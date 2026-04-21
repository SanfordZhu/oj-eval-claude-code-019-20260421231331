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

    // ---- 2) Append values[i] to v (stored in HBM between rounds) ----
    {
      Matrix *new_v = matrix_memory_allocator.Allocate("v_hbm");
      if (v_sram == nullptr) {
        gpu_sim.Copy(values[i], new_v, kInGpuHbm);
      } else {
        gpu_sim.Concat(v_sram, values[i], new_v, 0, kInGpuHbm);
        gpu_sim.ReleaseMatrix(v_sram);
      }
      gpu_sim.ReleaseMatrix(values[i]);
      v_sram = new_v;  // Currently in HBM; will be moved to SRAM before use.
    }

    // ---- 3) Q and kT both stay in HBM.  For each k we extract the column
    //        of Q and the row of kT in HBM, then move both to SRAM for the
    //        matmul.  This keeps Q*K^T peak SRAM tiny. ----

    // ---- 4) Compute QK = Q * K^T via outer-product split (n=1) ----
    Matrix *qk = nullptr;
    for (size_t k = 0; k < D; ++k) {
      Matrix *q_col = matrix_memory_allocator.Allocate("q_col");
      gpu_sim.GetColumn(current_query, k, q_col, kInGpuHbm);
      gpu_sim.MoveMatrixToSharedMem(q_col);

      Matrix *k_row = matrix_memory_allocator.Allocate("k_row");
      gpu_sim.GetRow(kT_hbm, k, k_row, kInGpuHbm);
      gpu_sim.MoveMatrixToSharedMem(k_row);

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

    // ---- 6) Compute attn column-by-column.  V stays in HBM; for each
    //        output column c:
    //          1) Extract column c of V (in HBM) and move to SRAM.
    //          2) out_col = softmax * v_col     (Rn x 1) in SRAM.
    //          3) Move out_col to HBM and concat into running attn_hbm.
    //        This keeps SRAM peak tiny (just softmax + a few columns). ----
    Matrix *attn_hbm = nullptr;
    for (size_t c = 0; c < D; ++c) {
      Matrix *v_col = matrix_memory_allocator.Allocate("v_col_hbm");
      gpu_sim.GetColumn(v_sram, c, v_col, kInGpuHbm);
      gpu_sim.MoveMatrixToSharedMem(v_col);

      Matrix *out_col = matrix_memory_allocator.Allocate("out_col");
      gpu_sim.MatMul(softmax, v_col, out_col);
      gpu_sim.ReleaseMatrix(v_col);

      gpu_sim.MoveMatrixToGpuHbm(out_col);

      if (attn_hbm == nullptr) {
        attn_hbm = out_col;
      } else {
        Matrix *new_attn = matrix_memory_allocator.Allocate("attn_hbm");
        gpu_sim.Concat(attn_hbm, out_col, new_attn, 1, kInGpuHbm);
        gpu_sim.ReleaseMatrix(attn_hbm);
        gpu_sim.ReleaseMatrix(out_col);
        attn_hbm = new_attn;
      }
    }
    gpu_sim.ReleaseMatrix(softmax);

    // ---- 7) attn is already in HBM; just commit ----
    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*attn_hbm);
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
