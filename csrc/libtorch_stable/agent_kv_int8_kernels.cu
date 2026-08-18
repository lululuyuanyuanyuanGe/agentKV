#include <torch/csrc/stable/tensor.h>

#include <cmath>
#include <cstdint>

#include "cub_helpers.h"
#include "dispatch_utils.h"
#include "ops.h"
#include "torch_utils.h"

namespace {

constexpr int kThreads = 256;

template <typename scalar_t>
__global__ void quantize_pack_kernel(
    const scalar_t* __restrict__ input, const int64_t* __restrict__ block_ids,
    uint8_t* __restrict__ packed, int64_t input_rows, int64_t input_stride,
    int64_t elements_per_block, int64_t packed_stride, int64_t payload_offset,
    int64_t scale_offset) {
  const int64_t packed_row = blockIdx.x;
  const int64_t input_row = block_ids[packed_row];
  if (input_row < 0 || input_row >= input_rows) {
    return;
  }

  const scalar_t* input_block = input + input_row * input_stride;
  float thread_absmax = 0.0f;
  for (int64_t index = threadIdx.x; index < elements_per_block;
       index += blockDim.x) {
    thread_absmax =
        fmaxf(thread_absmax, fabsf(static_cast<float>(input_block[index])));
  }

  using BlockReduce = cub::BlockReduce<float, kThreads>;
  __shared__ typename BlockReduce::TempStorage reduce_storage;
  __shared__ float scale;
  const float absmax =
      BlockReduce(reduce_storage).Reduce(thread_absmax, CubMaxOp{});
  if (threadIdx.x == 0) {
    scale = absmax > 0.0f ? absmax / 127.0f : 1.0f;
    uint8_t* packed_row_ptr = packed + packed_row * packed_stride;
    *reinterpret_cast<float*>(packed_row_ptr + scale_offset) = scale;
  }
  __syncthreads();

  int8_t* payload = reinterpret_cast<int8_t*>(
      packed + packed_row * packed_stride + payload_offset);
  const float inverse_scale = 1.0f / scale;
  for (int64_t index = threadIdx.x; index < elements_per_block;
       index += blockDim.x) {
    const float value = static_cast<float>(input_block[index]) * inverse_scale;
    const int rounded = __float2int_rn(value);
    const int quantized = rounded < -127 ? -127 : rounded > 127 ? 127 : rounded;
    payload[index] = static_cast<int8_t>(quantized);
  }
}

template <typename scalar_t>
__global__ void unpack_dequantize_kernel(
    const uint8_t* __restrict__ packed, const int64_t* __restrict__ block_ids,
    scalar_t* __restrict__ output, int64_t output_rows, int64_t output_stride,
    int64_t elements_per_block, int64_t packed_stride, int64_t payload_offset,
    int64_t scale_offset) {
  const int64_t packed_row = blockIdx.x;
  const int64_t output_row = block_ids[packed_row];
  if (output_row < 0 || output_row >= output_rows) {
    return;
  }

  const uint8_t* packed_row_ptr = packed + packed_row * packed_stride;
  const float scale =
      *reinterpret_cast<const float*>(packed_row_ptr + scale_offset);
  const int8_t* payload =
      reinterpret_cast<const int8_t*>(packed_row_ptr + payload_offset);
  scalar_t* output_block = output + output_row * output_stride;
  for (int64_t index = threadIdx.x; index < elements_per_block;
       index += blockDim.x) {
    output_block[index] =
        static_cast<scalar_t>(static_cast<float>(payload[index]) * scale);
  }
}

void validate_common(const torch::stable::Tensor& values,
                     const torch::stable::Tensor& block_ids,
                     const torch::stable::Tensor& packed_buffer,
                     int64_t payload_offset, int64_t scale_offset) {
  STD_TORCH_CHECK(values.device().is_cuda());
  STD_TORCH_CHECK(block_ids.device().is_cuda());
  STD_TORCH_CHECK(packed_buffer.device().is_cuda());
  STD_TORCH_CHECK(values.device() == block_ids.device());
  STD_TORCH_CHECK(values.device() == packed_buffer.device());
  STD_TORCH_CHECK(values.dim() == 2);
  STD_TORCH_CHECK(block_ids.dim() == 1);
  STD_TORCH_CHECK(packed_buffer.dim() == 2);
  STD_TORCH_CHECK(values.is_contiguous());
  STD_TORCH_CHECK(block_ids.is_contiguous());
  STD_TORCH_CHECK(packed_buffer.is_contiguous());
  STD_TORCH_CHECK(block_ids.scalar_type() ==
                  torch::headeronly::ScalarType::Long);
  STD_TORCH_CHECK(packed_buffer.scalar_type() ==
                  torch::headeronly::ScalarType::Byte);
  STD_TORCH_CHECK(block_ids.numel() <= packed_buffer.size(0));
  STD_TORCH_CHECK(packed_buffer.stride(0) % alignof(float) == 0);
  STD_TORCH_CHECK(reinterpret_cast<std::uintptr_t>(
                      packed_buffer.const_data_ptr<uint8_t>()) %
                      alignof(float) ==
                  0);
  STD_TORCH_CHECK(payload_offset >= 0);
  STD_TORCH_CHECK(scale_offset >= 0 && scale_offset % alignof(float) == 0);
  STD_TORCH_CHECK(payload_offset + values.size(1) <= packed_buffer.size(1));
  STD_TORCH_CHECK(scale_offset + static_cast<int64_t>(sizeof(float)) <=
                  packed_buffer.size(1));
  STD_TORCH_CHECK(payload_offset >=
                      scale_offset + static_cast<int64_t>(sizeof(float)) ||
                  scale_offset >= payload_offset + values.size(1));
}

}  // namespace

void agent_kv_int8_quantize_pack(const torch::stable::Tensor& input,
                                 const torch::stable::Tensor& block_ids,
                                 torch::stable::Tensor& packed_buffer,
                                 int64_t payload_offset, int64_t scale_offset) {
  validate_common(input, block_ids, packed_buffer, payload_offset,
                  scale_offset);
  if (block_ids.numel() == 0) {
    return;
  }

  const torch::stable::accelerator::DeviceGuard device_guard(
      input.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();
  const dim3 grid(static_cast<unsigned int>(block_ids.numel()));
  const dim3 block(kThreads);
  VLLM_STABLE_DISPATCH_HALF_TYPES(
      input.scalar_type(), "agent_kv_int8_quantize_pack", [&] {
        quantize_pack_kernel<scalar_t><<<grid, block, 0, stream>>>(
            input.const_data_ptr<scalar_t>(),
            block_ids.const_data_ptr<int64_t>(),
            packed_buffer.mutable_data_ptr<uint8_t>(), input.size(0),
            input.stride(0), input.size(1), packed_buffer.stride(0),
            payload_offset, scale_offset);
      });
  STD_CUDA_CHECK(cudaGetLastError());
}

void agent_kv_int8_unpack_dequantize(const torch::stable::Tensor& packed_buffer,
                                     const torch::stable::Tensor& block_ids,
                                     torch::stable::Tensor& output,
                                     int64_t payload_offset,
                                     int64_t scale_offset) {
  validate_common(output, block_ids, packed_buffer, payload_offset,
                  scale_offset);
  if (block_ids.numel() == 0) {
    return;
  }

  const torch::stable::accelerator::DeviceGuard device_guard(
      output.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();
  const dim3 grid(static_cast<unsigned int>(block_ids.numel()));
  const dim3 block(kThreads);
  VLLM_STABLE_DISPATCH_HALF_TYPES(
      output.scalar_type(), "agent_kv_int8_unpack_dequantize", [&] {
        unpack_dequantize_kernel<scalar_t><<<grid, block, 0, stream>>>(
            packed_buffer.const_data_ptr<uint8_t>(),
            block_ids.const_data_ptr<int64_t>(),
            output.mutable_data_ptr<scalar_t>(), output.size(0),
            output.stride(0), output.size(1), packed_buffer.stride(0),
            payload_offset, scale_offset);
      });
  STD_CUDA_CHECK(cudaGetLastError());
}
