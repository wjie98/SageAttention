/*
 * Copyright (c) 2024 by SageAttention team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/extension.h>

void quant_per_block_int8_cuda(
                torch::Tensor input,
                torch::Tensor output,
                torch::Tensor scale,
                float sm_scale,
                int block_size,
                int tensor_layout);

void quant_per_block_int8_cuda(
                torch::Tensor input,
                torch::Tensor output,
                torch::Tensor scale,
                int block_size,
                int tensor_layout);

void quant_per_block_int8_fuse_sub_mean_cuda(
                torch::Tensor input,
                torch::Tensor mean,
                torch::Tensor output,
                torch::Tensor scale,
                int block_size,
                int tensor_layout);

void quant_per_warp_int8_cuda(
                torch::Tensor input,
                torch::Tensor output,
                torch::Tensor scale,
                int block_size,
                int warp_block_size,
                int tensor_layout);

void quant_qk_per_warp_int8_cuda(
                torch::Tensor query,
                torch::Tensor key,
                torch::Tensor query_output,
                torch::Tensor key_output,
                torch::Tensor query_scale,
                torch::Tensor key_scale,
                int query_block_size,
                int query_warp_block_size,
                int key_block_size,
                int tensor_layout);

void quant_per_warp_int8_varlen_cuda(
                torch::Tensor input,
                torch::Tensor cu_seqlens,
                torch::Tensor output,
                torch::Tensor scale,
                int max_seqlen,
                int block_size,
                int warp_block_size);

void sub_mean_cuda(
                torch::Tensor input,
                torch::Tensor mean,
                torch::Tensor output,
                int tensor_layout);

void transpose_pad_permute_cuda(
                torch::Tensor input,
                torch::Tensor output,
                int tensor_layout);

void scale_fuse_quant_cuda(
                torch::Tensor input,
                torch::Tensor output,
                torch::Tensor scale,
                int num_tokens,
                float scale_max,
                int tensor_layout);

void mean_scale_fuse_quant_cuda(
                torch::Tensor input,
                torch::Tensor output,
                torch::Tensor mean,
                torch::Tensor scale,
                int num_tokens,
                float scale_max,
                int tensor_layout);

void varlen_attention_fwd_cuda(
                torch::Tensor query,
                torch::Tensor key,
                torch::Tensor value,
                torch::Tensor cu_seqlens_q,
                torch::Tensor cu_seqlens_k,
                torch::Tensor output,
                int max_seqlen_q,
                float sm_scale,
                int is_causal);
