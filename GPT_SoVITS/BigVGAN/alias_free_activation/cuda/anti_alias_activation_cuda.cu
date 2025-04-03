/* coding=utf-8
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include "type_shim.h"
#include <assert.h>
#include <cfloat>
#include <limits>
#include <stdint.h>
#include <c10/macros/Macros.h>

namespace
{
    // Hard-coded hyperparameters
    // WARP_SIZE and WARP_BATCH must match the return values batches_per_warp and
    constexpr int ELEMENTS_PER_LDG_STG = 1; //(WARP_ITERATIONS < 4) ? 1 : 4;
    constexpr int BUFFER_SIZE = 32;
    constexpr int FILTER_SIZE = 12;
    constexpr int HALF_FILTER_SIZE = 6;
    constexpr int UPSAMPLE_REPLICATION_PAD = 5; // 5 on each side, matching torch impl
    constexpr int DOWNSAMPLE_REPLICATION_PAD_LEFT = 5; // matching torch impl
    constexpr int DOWNSAMPLE_REPLICATION_PAD_RIGHT = 6; // matching torch impl

    template <typename input_t, typename output_t, typename acc_t>
    __global__ void anti_alias_activation_forward(
        output_t *dst,
        const input_t *src,
        const input_t *up_ftr,
        const input_t *down_ftr,
        const input_t *alpha,
        const input_t *beta,
        int batch_size,
        int channels,
        int seq_len)
    {
        // Up and downsample filters
        input_t up_filter[FILTER_SIZE];
        input_t down_filter[FILTER_SIZE];

        // Load data from global memory including extra indices reserved for replication paddings
        input_t elements[2 * FILTER_SIZE + 2 * BUFFER_SIZE + 2 * UPSAMPLE_REPLICATION_PAD] = {0};
        input_t intermediates[2 * FILTER_SIZE + 2 * BUFFER_SIZE + DOWNSAMPLE_REPLICATION_PAD_LEFT + DOWNSAMPLE_REPLICATION_PAD_RIGHT] = {0};

        // Output stores downsampled output before writing to dst
        output_t output[BUFFER_SIZE];

        // blockDim/threadIdx = (128, 1, 1)
        // gridDim/blockIdx = (seq_blocks, channels, batches)
        int block_offset = (blockIdx.x * 128 * BUFFER_SIZE + seq_len * (blockIdx.y + gridDim.y * blockIdx.z));
        int local_offset = threadIdx.x * BUFFER_SIZE;
        int seq_offset = blockIdx.x * 128 * BUFFER_SIZE + local_offset;

        // intermediate have double the seq_len
        int intermediate_local_offset = threadIdx.x * BUFFER_SIZE * 2;
        int intermediate_seq_offset = blockIdx.x * 128 * BUFFER_SIZE * 2 + intermediate_local_offset;

        // Get values needed for replication padding before moving pointer
        const input_t *right_most_pntr = src + (seq_len * (blockIdx.y + gridDim.y * blockIdx.z));
        input_t seq_left_most_value = right_most_pntr[0];
        input_t seq_right_most_value = right_most_pntr[seq_len - 1];

        // Move src and dst pointers
        src += block_offset + local_offset;
        dst += block_offset + local_offset;

        // Alpha and beta values for snake activatons. Applies exp by default
        alpha = alpha + blockIdx.y;
        input_t alpha_val = expf(alpha[0]);
        beta = beta + blockIdx.y;
        input_t beta_val = expf(beta[0]);

        #pragma unroll
        for (int it = 0; it < FILTER_SIZE; it += 1)
        {
            up_filter[it] = up_ftr[it];
            down_filter[it] = down_ftr[it];
        }

        // Apply replication padding for upsampling, matching torch impl
        #pragma unroll
        for (int it = -HALF_FILTER_SIZE; it < BUFFER_SIZE + HALF_FILTER_SIZE; it += 1)
        {
            int element_index = seq_offset + it; // index for element
            if ((element_index < 0) && (element_index >= -UPSAMPLE_REPLICATION_PAD))
            {
                elements[2 * (HALF_FILTER_SIZE + it)] = 2 * seq_left_most_value;
            }
            if ((element_index >= seq_len) && (element_index < seq_len + UPSAMPLE_REPLICATION_PAD))
            {
                elements[2 * (HALF_FILTER_SIZE + it)] = 2 * seq_right_most_value;
            }
            if ((element_index >= 0) && (element_index < seq_len))
            {
                elements[2 * (HALF_FILTER_SIZE + it)] = 2 * src[it];
            }
        }

        // Apply upsampling strided convolution and write to intermediates. It reserves DOWNSAMPLE_REPLICATION_PAD_LEFT for replication padding of the downsampilng conv later
        #pragma unroll
        for (int it = 0; it < (2 * BUFFER_SIZE + 2 * FILTER_SIZE); it += 1)
        {
            input_t acc = 0.0;
            int element_index = intermediate_seq_offset + it; // index for intermediate
            #pragma unroll
            for (int f_idx = 0; f_idx < FILTER_SIZE; f_idx += 1)
            {
                if ((element_index + f_idx) >= 0)
                {
                    acc += up_filter[f_idx] * elements[it + f_idx];
                }
            }
            intermediates[it + DOWNSAMPLE_REPLICATION_PAD_LEFT] = acc;
        }

        // Apply activation function. It reserves DOWNSAMPLE_REPLICATION_PAD_LEFT and DOWNSAMPLE_REPLICATION_PAD_RIGHT for replication padding of the downsampilng conv later
        double no_div_by_zero = 0.000000001;
        #pragma unroll
        for (int it = 0; it < 2 * BUFFER_SIZE + 2 * FILTER_SIZE; it += 1)
        {
            intermediates[it + DOWNSAMPLE_REPLICATION_PAD_LEFT] += (1.0 / (beta_val + no_div_by_zero)) * sinf(intermediates[it + DOWNSAMPLE_REPLICATION_PAD_LEFT] * alpha_val) * sinf(intermediates[it + DOWNSAMPLE_REPLICATION_PAD_LEFT] * alpha_val);
        }

        // Apply replication padding before downsampling conv from intermediates
        #pragma unroll
        for (int it = 0; it < DOWNSAMPLE_REPLICATION_PAD_LEFT; it += 1)
        {
            intermediates[it] = intermediates[DOWNSAMPLE_REPLICATION_PAD_LEFT];
        }
        #pragma unroll
        for (int it = DOWNSAMPLE_REPLICATION_PAD_LEFT + 2 * BUFFER_SIZE + 2 * FILTER_SIZE; it < DOWNSAMPLE_REPLICATION_PAD_LEFT + 2 * BUFFER_SIZE + 2 * FILTER_SIZE + DOWNSAMPLE_REPLICATION_PAD_RIGHT; it += 1)
        {
            intermediates[it] = intermediates[DOWNSAMPLE_REPLICATION_PAD_LEFT + 2 * BUFFER_SIZE + 2 * FILTER_SIZE - 1];
        }

        // Apply downsample strided convolution (assuming stride=2) from intermediates
        #pragma unroll
        for (int it = 0; it < BUFFER_SIZE; it += 1)
        {
            input_t acc = 0.0;
            #pragma unroll
            for (int f_idx = 0; f_idx < FILTER_SIZE; f_idx += 1)
            {
                // Add constant DOWNSAMPLE_REPLICATION_PAD_RIGHT to match torch implementation
                acc += down_filter[f_idx] * intermediates[it * 2 + f_idx + DOWNSAMPLE_REPLICATION_PAD_RIGHT];
            }
            output[it] = acc;
        }

        // Write output to dst
        #pragma unroll
        for (int it = 0;  it < BUFFER_SIZE;  it += ELEMENTS_PER_LDG_STG)
        {
            int element_index = seq_offset + it;
            if (element_index < seq_len)
            {
                dst[it] = output[it];
            }
        }

    }

    template <typename input_t, typename output_t, typename acc_t>
    void dispatch_anti_alias_activation_forward(
        output_t *dst,
        const input_t *src,
        const input_t *up_ftr,
        const input_t *down_ftr,
        const input_t *alpha,
        const input_t *beta,
        int batch_size,
        int channels,
        int seq_len)
    {
        if (seq_len == 0)
        {
            return;
        }
        else
        {
            // Use 128 threads per block to maximimize gpu utilization
            constexpr int threads_per_block = 128;
            constexpr int seq_len_per_block = 4096;
            int blocks_per_seq_len = (seq_len + seq_len_per_block - 1) / seq_len_per_block;
            dim3 blocks(blocks_per_seq_len, channels, batch_size);
            dim3 threads(threads_per_block, 1, 1);

            anti_alias_activation_forward<input_t, output_t, acc_t>
                <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(dst, src, up_ftr, down_ftr, alpha, beta, batch_size, channels, seq_len);
        }
    }
}

extern "C" torch::Tensor fwd_cuda(torch::Tensor const &input, torch::Tensor const &up_filter, torch::Tensor const &down_filter, torch::Tensor const &alpha, torch::Tensor const &beta)
{
    // Input is a 3d tensor with dimensions [batches, channels, seq_len]
    const int batches = input.size(0);
    const int channels = input.size(1);
    const int seq_len = input.size(2);

    // Output
    auto act_options = input.options().requires_grad(false);

    torch::Tensor anti_alias_activation_results =
        torch::empty({batches, channels, seq_len}, act_options);

    void *input_ptr = static_cast<void *>(input.data_ptr());
    void *up_filter_ptr = static_cast<void *>(up_filter.data_ptr());
    void *down_filter_ptr = static_cast<void *>(down_filter.data_ptr());
    void *alpha_ptr = static_cast<void *>(alpha.data_ptr());
    void *beta_ptr = static_cast<void *>(beta.data_ptr());
    void *anti_alias_activation_results_ptr = static_cast<void *>(anti_alias_activation_results.data_ptr());

    DISPATCH_FLOAT_HALF_AND_BFLOAT(
        input.scalar_type(),
        "dispatch anti alias activation_forward",
        dispatch_anti_alias_activation_forward<scalar_t, scalar_t, float>(
            reinterpret_cast<scalar_t *>(anti_alias_activation_results_ptr),
            reinterpret_cast<const scalar_t *>(input_ptr),
            reinterpret_cast<const scalar_t *>(up_filter_ptr),
            reinterpret_cast<const scalar_t *>(down_filter_ptr),
            reinterpret_cast<const scalar_t *>(alpha_ptr),
            reinterpret_cast<const scalar_t *>(beta_ptr),
            batches,
            channels,
            seq_len););
    return anti_alias_activation_results;
}