#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());

  // Persistent: kT in HBM, v in SRAM.
  Matrix *kT_hbm = nullptr;
  Matrix *v_sram = nullptr;

  const size_t D = 512;

  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();
    const size_t Rn = i + 1;

    // ---- 1) Append keys[i] to kT_hbm (in HBM) ----
    gpu_sim.Transpose(keys[i], kInGpuHbm);
    {
      Matrix *new_kT = matrix_memory_allocator.Allocate("kT_hbm");
      if (kT_hbm == nullptr) {
        gpu_sim.Copy(keys[i], new_kT, kInGpuHbm);
      } else {
        gpu_sim.Concat(kT_hbm, keys[i], new_kT, 1, kInGpuHbm);
        gpu_sim.ReleaseMatrix(kT_hbm);
      }
      gpu_sim.ReleaseMatrix(keys[i]);
      kT_hbm = new_kT;
    }

    // ---- 2) Append values[i] to v_sram (SRAM) ----
    gpu_sim.MoveMatrixToSharedMem(values[i]);
    {
      Matrix *new_v = matrix_memory_allocator.Allocate("v_sram");
      if (v_sram == nullptr) {
        gpu_sim.Copy(values[i], new_v, kInSharedMemory);
      } else {
        gpu_sim.Concat(v_sram, values[i], new_v, 0, kInSharedMemory);
        gpu_sim.ReleaseMatrix(v_sram);
      }
      gpu_sim.ReleaseMatrix(values[i]);
      v_sram = new_v;
    }

    // ---- 3) Move Q and kT to SRAM for Q*K^T ----
    gpu_sim.MoveMatrixToSharedMem(current_query);
    Matrix *kT_sram = matrix_memory_allocator.Allocate("kT_sram");
    gpu_sim.Copy(kT_hbm, kT_sram, kInGpuHbm);
    gpu_sim.MoveMatrixToSharedMem(kT_sram);

    // ---- 4) Compute QK = Q * K^T via outer-product split (n=1) ----
    Matrix *qk = nullptr;
    for (size_t k = 0; k < D; ++k) {
      Matrix *q_col = matrix_memory_allocator.Allocate("q_col");
      gpu_sim.GetColumn(current_query, k, q_col, kInSharedMemory);
      Matrix *k_row = matrix_memory_allocator.Allocate("k_row");
      gpu_sim.GetRow(kT_sram, k, k_row, kInSharedMemory);
      Matrix *outer = matrix_memory_allocator.Allocate("outer");
      gpu_sim.MatMul(q_col, k_row, outer);
      gpu_sim.ReleaseMatrix(q_col);
      gpu_sim.ReleaseMatrix(k_row);
      if (qk == nullptr) {
        qk = outer;
      } else {
        Matrix *new_qk = matrix_memory_allocator.Allocate("qk");
        gpu_sim.MatAdd(qk, outer, new_qk);
        gpu_sim.ReleaseMatrix(qk);
        gpu_sim.ReleaseMatrix(outer);
        qk = new_qk;
      }
    }
    gpu_sim.ReleaseMatrix(current_query);
    gpu_sim.ReleaseMatrix(kT_sram);

    // ---- 5) Row-wise softmax on qk ----
    Matrix *exp_qk = matrix_memory_allocator.Allocate("exp_qk");
    gpu_sim.MatExp(qk, exp_qk);
    gpu_sim.ReleaseMatrix(qk);

    Matrix *softmax = nullptr;
    for (size_t r = 0; r < Rn; ++r) {
      Matrix *row = matrix_memory_allocator.Allocate("row");
      gpu_sim.GetRow(exp_qk, r, row, kInSharedMemory);
      Matrix *rs = matrix_memory_allocator.Allocate("rs");
      gpu_sim.Sum(row, rs);
      Matrix *rsm = matrix_memory_allocator.Allocate("rsm");
      gpu_sim.MatDiv(row, rs, rsm);
      gpu_sim.ReleaseMatrix(row);
      gpu_sim.ReleaseMatrix(rs);
      if (softmax == nullptr) {
        softmax = rsm;
      } else {
        Matrix *new_sm = matrix_memory_allocator.Allocate("sm");
        gpu_sim.Concat(softmax, rsm, new_sm, 0, kInSharedMemory);
        gpu_sim.ReleaseMatrix(softmax);
        gpu_sim.ReleaseMatrix(rsm);
        softmax = new_sm;
      }
    }
    gpu_sim.ReleaseMatrix(exp_qk);

    // ---- 6) Compute attn row-by-row, concatenating rows to build attn.
    //        For each output row r:
    //          row_r (1, D) = sum_k softmax[r, k] * V[k, :]
    //        Then concat row_r to running attn (axis=0). ----
    Matrix *attn = nullptr;
    for (size_t r = 0; r < Rn; ++r) {
      // Extract softmax row r as (1, Rn)
      Matrix *sm_row = matrix_memory_allocator.Allocate("sm_row");
      gpu_sim.GetRow(softmax, r, sm_row, kInSharedMemory);

      // Build row_r by summing scaled V rows.
      Matrix *row_acc = nullptr;
      for (size_t k = 0; k < Rn; ++k) {
        Matrix *scalar = matrix_memory_allocator.Allocate("scalar");
        gpu_sim.GetColumn(sm_row, k, scalar, kInSharedMemory);  // (1,1)
        Matrix *v_row = matrix_memory_allocator.Allocate("v_row");
        gpu_sim.GetRow(v_sram, k, v_row, kInSharedMemory);  // (1, D)
        Matrix *scaled = matrix_memory_allocator.Allocate("scaled");
        // MulNum computes scaled = v_row * scalar
        // But gpu_sim has no MulNum in instruction API... check.
        // Actually gpu_sim.h doesn't expose MulNum. Use MatMul with a 1x1.
        // Wait, MatMul(A(1,1), B(1,D)) yields (1, D). Cost = 5*1*1*D = 5D.
        // That's same as MulNum(4D) roughly.
        gpu_sim.MatMul(scalar, v_row, scaled);
        gpu_sim.ReleaseMatrix(scalar);
        gpu_sim.ReleaseMatrix(v_row);
        if (row_acc == nullptr) {
          row_acc = scaled;
        } else {
          Matrix *new_row_acc = matrix_memory_allocator.Allocate("row_acc");
          gpu_sim.MatAdd(row_acc, scaled, new_row_acc);
          gpu_sim.ReleaseMatrix(row_acc);
          gpu_sim.ReleaseMatrix(scaled);
          row_acc = new_row_acc;
        }
      }
      gpu_sim.ReleaseMatrix(sm_row);

      // Concat row_acc into final attn (axis=0).
      if (attn == nullptr) {
        attn = row_acc;
      } else {
        Matrix *new_attn = matrix_memory_allocator.Allocate("attn");
        gpu_sim.Concat(attn, row_acc, new_attn, 0, kInSharedMemory);
        gpu_sim.ReleaseMatrix(attn);
        gpu_sim.ReleaseMatrix(row_acc);
        attn = new_attn;
      }
    }
    gpu_sim.ReleaseMatrix(softmax);

    // ---- 7) Move attn to HBM and commit ----
    gpu_sim.MoveMatrixToGpuHbm(attn);
    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*attn);
  }

  if (kT_hbm != nullptr) gpu_sim.ReleaseMatrix(kT_hbm);
  if (v_sram != nullptr) gpu_sim.ReleaseMatrix(v_sram);
  gpu_sim.Run(false, &matrix_memory_allocator);
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu
