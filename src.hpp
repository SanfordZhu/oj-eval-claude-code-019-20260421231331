#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());

  // Accumulated K^T (shape 512 x (i+1)) and V (shape (i+1) x 512) in SRAM.
  Matrix *kT_sram = nullptr;
  Matrix *v_sram = nullptr;

  const size_t D = 512;  // feature dimension

  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();
    const size_t Rn = i + 1;  // number of rows in Q

    // ---- 1) Move current key/value to SRAM and accumulate ----
    gpu_sim.MoveMatrixToSharedMem(keys[i]);
    gpu_sim.MoveMatrixToSharedMem(values[i]);

    // Transpose key (1xD) -> (Dx1), in SRAM (cost D)
    gpu_sim.Transpose(keys[i], kInSharedMemory);

    // Append to kT_sram along axis=1  -> shape (D, Rn)
    {
      Matrix *new_kT = matrix_memory_allocator.Allocate("kT");
      if (kT_sram == nullptr) {
        gpu_sim.Copy(keys[i], new_kT, kInSharedMemory);
      } else {
        gpu_sim.Concat(kT_sram, keys[i], new_kT, 1, kInSharedMemory);
        gpu_sim.ReleaseMatrix(kT_sram);
      }
      gpu_sim.ReleaseMatrix(keys[i]);
      kT_sram = new_kT;
    }

    // Append to v_sram along axis=0  -> shape (Rn, D)
    {
      Matrix *new_v = matrix_memory_allocator.Allocate("v");
      if (v_sram == nullptr) {
        gpu_sim.Copy(values[i], new_v, kInSharedMemory);
      } else {
        gpu_sim.Concat(v_sram, values[i], new_v, 0, kInSharedMemory);
        gpu_sim.ReleaseMatrix(v_sram);
      }
      gpu_sim.ReleaseMatrix(values[i]);
      v_sram = new_v;
    }

    // ---- 2) Move Q to SRAM ----
    gpu_sim.MoveMatrixToSharedMem(current_query);

    // ---- 3) Compute QK = Q * K^T via outer-product split ----
    // Q is (Rn, D), K^T is (D, Rn). For each k in 0..D-1:
    //   outer = Q[:,k:k+1] * K^T[k:k+1,:]  which is (Rn x Rn).
    // Sum all outers to get QK.
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

    // ---- 4) Row-wise softmax on qk ----
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

    // ---- 5) Compute attn = softmax * V via outer-product split ----
    // softmax is (Rn, Rn), V is (Rn, D). For each k in 0..Rn-1:
    //   outer = softmax[:,k:k+1] * V[k:k+1,:]  which is (Rn x D).
    Matrix *attn = nullptr;
    for (size_t k = 0; k < Rn; ++k) {
      Matrix *s_col = matrix_memory_allocator.Allocate("s_col");
      gpu_sim.GetColumn(softmax, k, s_col, kInSharedMemory);
      Matrix *v_row = matrix_memory_allocator.Allocate("v_row");
      gpu_sim.GetRow(v_sram, k, v_row, kInSharedMemory);
      Matrix *outer = matrix_memory_allocator.Allocate("vouter");
      gpu_sim.MatMul(s_col, v_row, outer);
      gpu_sim.ReleaseMatrix(s_col);
      gpu_sim.ReleaseMatrix(v_row);
      if (attn == nullptr) {
        attn = outer;
      } else {
        Matrix *new_attn = matrix_memory_allocator.Allocate("attn");
        gpu_sim.MatAdd(attn, outer, new_attn);
        gpu_sim.ReleaseMatrix(attn);
        gpu_sim.ReleaseMatrix(outer);
        attn = new_attn;
      }
    }
    gpu_sim.ReleaseMatrix(softmax);

    // ---- 6) Move attn to HBM and commit ----
    gpu_sim.MoveMatrixToGpuHbm(attn);
    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*attn);
  }

  if (kT_sram != nullptr) gpu_sim.ReleaseMatrix(kT_sram);
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
