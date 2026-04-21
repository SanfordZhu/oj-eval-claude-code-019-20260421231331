#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());

  // Accumulated K^T (shape 512 x (i+1)) and V (shape (i+1) x 512) in SRAM.
  Matrix *kT_sram = nullptr; // K_transposed concatenated, in SRAM
  Matrix *v_sram = nullptr;  // V concatenated, in SRAM

  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();

    // 1. Move current key and value into SRAM.
    gpu_sim.MoveMatrixToSharedMem(keys[i]);
    gpu_sim.MoveMatrixToSharedMem(values[i]);

    // 2. Transpose the new key so it becomes (512 x 1).
    gpu_sim.Transpose(keys[i], kInSharedMemory);

    // 3. Append to accumulated K^T (concat along axis=1, i.e. append column).
    Matrix *new_kT = matrix_memory_allocator.Allocate("kT_" + std::to_string(i));
    if (kT_sram == nullptr) {
      // First round: just copy keys[i] to new_kT (512x1)
      gpu_sim.Copy(keys[i], new_kT, kInSharedMemory);
    } else {
      gpu_sim.Concat(kT_sram, keys[i], new_kT, 1, kInSharedMemory);
      gpu_sim.ReleaseMatrix(kT_sram);
    }
    // Release the transposed single key (we've copied/concatenated it).
    gpu_sim.ReleaseMatrix(keys[i]);
    kT_sram = new_kT;

    // 4. Append values[i] as new row of V (axis=0).
    Matrix *new_v = matrix_memory_allocator.Allocate("v_" + std::to_string(i));
    if (v_sram == nullptr) {
      gpu_sim.Copy(values[i], new_v, kInSharedMemory);
    } else {
      gpu_sim.Concat(v_sram, values[i], new_v, 0, kInSharedMemory);
      gpu_sim.ReleaseMatrix(v_sram);
    }
    gpu_sim.ReleaseMatrix(values[i]);
    v_sram = new_v;

    // 5. Move Q (current_query, shape (i+1, 512)) to SRAM.
    gpu_sim.MoveMatrixToSharedMem(current_query);

    // 6. Compute QK = Q * K^T  -> shape (i+1, i+1)
    Matrix *qk = matrix_memory_allocator.Allocate("qk_" + std::to_string(i));
    gpu_sim.MatMul(current_query, kT_sram, qk);
    // We don't need Q in SRAM anymore; but we shouldn't release until matmul done.
    gpu_sim.ReleaseMatrix(current_query);

    // 7. Row-wise softmax.
    // Apply exp to whole matrix.
    Matrix *exp_qk = matrix_memory_allocator.Allocate("exp_qk_" + std::to_string(i));
    gpu_sim.MatExp(qk, exp_qk);
    gpu_sim.ReleaseMatrix(qk);

    // For each row, get row sum and divide; concat divided rows.
    // Then compute attn * V = output row.
    // Build full softmax matrix row by row:
    Matrix *softmax = nullptr;
    for (size_t r = 0; r <= i; ++r) {
      Matrix *row = matrix_memory_allocator.Allocate("row_" + std::to_string(r));
      gpu_sim.GetRow(exp_qk, r, row, kInSharedMemory);
      Matrix *row_sum = matrix_memory_allocator.Allocate("rs_" + std::to_string(r));
      gpu_sim.Sum(row, row_sum);
      Matrix *row_softmax = matrix_memory_allocator.Allocate("rsm_" + std::to_string(r));
      gpu_sim.MatDiv(row, row_sum, row_softmax);
      gpu_sim.ReleaseMatrix(row);
      gpu_sim.ReleaseMatrix(row_sum);
      if (softmax == nullptr) {
        softmax = row_softmax;
      } else {
        Matrix *new_softmax = matrix_memory_allocator.Allocate("sm_" + std::to_string(r));
        gpu_sim.Concat(softmax, row_softmax, new_softmax, 0, kInSharedMemory);
        gpu_sim.ReleaseMatrix(softmax);
        gpu_sim.ReleaseMatrix(row_softmax);
        softmax = new_softmax;
      }
    }
    gpu_sim.ReleaseMatrix(exp_qk);

    // 8. attn = softmax * V   shape = (i+1, 512)
    Matrix *attn = matrix_memory_allocator.Allocate("attn_" + std::to_string(i));
    gpu_sim.MatMul(softmax, v_sram, attn);
    gpu_sim.ReleaseMatrix(softmax);

    // 9. Move attn to HBM and commit.
    gpu_sim.MoveMatrixToGpuHbm(attn);

    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*attn);
  }

  // Release accumulated SRAM matrices.
  if (kT_sram != nullptr) {
    gpu_sim.ReleaseMatrix(kT_sram);
  }
  if (v_sram != nullptr) {
    gpu_sim.ReleaseMatrix(v_sram);
  }
  gpu_sim.Run(false, &matrix_memory_allocator);
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu
