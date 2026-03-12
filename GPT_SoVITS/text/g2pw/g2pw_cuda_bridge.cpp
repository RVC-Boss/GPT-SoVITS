#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "g2pw/runtime.h"

namespace {

struct G2PWRuntimeHandle {
  std::unique_ptr<g2pw::Runtime> runtime;
  std::string last_error;
  int num_labels = 0;
};

void SetError(G2PWRuntimeHandle* handle, const g2pw::Status& status) {
  if (handle == nullptr) {
    return;
  }
  handle->last_error = status.message;
}

g2pw::RuntimeConfig BuildConfig(
    int device_ordinal,
    int max_batch_size,
    int max_seq_len,
    int full_graph_cache_limit,
    int tail_graph_cache_limit,
    int allow_tensor_cores,
    int use_cublaslt_bias_epilogue,
    int enable_profiling,
    int enable_cuda_graph,
    int dump_graph_cache_stats,
    int gemm_precision) {
  g2pw::RuntimeConfig config{};
  config.device_ordinal = device_ordinal;
  config.max_batch_size = max_batch_size;
  config.max_seq_len = max_seq_len;
  config.full_graph_cache_limit = full_graph_cache_limit;
  config.tail_graph_cache_limit = tail_graph_cache_limit;
  config.allow_tensor_cores = allow_tensor_cores != 0;
  config.use_cublaslt_bias_epilogue = use_cublaslt_bias_epilogue != 0;
  config.enable_profiling = enable_profiling != 0;
  config.enable_cuda_graph = enable_cuda_graph != 0;
  config.dump_graph_cache_stats = dump_graph_cache_stats != 0;
  switch (gemm_precision) {
    case 1:
      config.gemm_precision = g2pw::GemmPrecision::kFp16;
      break;
    case 2:
      config.gemm_precision = g2pw::GemmPrecision::kBf16;
      break;
    default:
      config.gemm_precision = g2pw::GemmPrecision::kFp32;
      break;
  }
  return config;
}

}  // namespace

extern "C" {

void* g2pw_runtime_create(
    const char* manifest_path,
    const char* binary_path,
    int device_ordinal,
    int max_batch_size,
    int max_seq_len,
    int full_graph_cache_limit,
    int tail_graph_cache_limit,
    int allow_tensor_cores,
    int use_cublaslt_bias_epilogue,
    int enable_profiling,
    int enable_cuda_graph,
    int dump_graph_cache_stats,
    int gemm_precision) {
  auto* handle = new G2PWRuntimeHandle();
  try {
    if (manifest_path == nullptr || binary_path == nullptr) {
      handle->last_error = "manifest_path and binary_path must be non-null";
      return handle;
    }
    g2pw::RuntimeConfig config = BuildConfig(
        device_ordinal,
        max_batch_size,
        max_seq_len,
        full_graph_cache_limit,
        tail_graph_cache_limit,
        allow_tensor_cores,
        use_cublaslt_bias_epilogue,
        enable_profiling,
        enable_cuda_graph,
        dump_graph_cache_stats,
        gemm_precision);
    g2pw::Status status = g2pw::Runtime::Create(
        config,
        std::string(manifest_path),
        std::string(binary_path),
        &handle->runtime);
    if (!status.ok()) {
      SetError(handle, status);
      return handle;
    }
    handle->num_labels = handle->runtime != nullptr ? handle->runtime->weights().manifest().num_labels : 0;
    handle->last_error.clear();
    return handle;
  } catch (const std::exception& exc) {
    handle->last_error = exc.what();
    return handle;
  } catch (...) {
    handle->last_error = "unknown exception";
    return handle;
  }
}

void g2pw_runtime_destroy(void* raw_handle) {
  auto* handle = static_cast<G2PWRuntimeHandle*>(raw_handle);
  delete handle;
}

const char* g2pw_runtime_last_error(void* raw_handle) {
  auto* handle = static_cast<G2PWRuntimeHandle*>(raw_handle);
  if (handle == nullptr) {
    return "invalid runtime handle";
  }
  return handle->last_error.c_str();
}

int g2pw_runtime_num_labels(void* raw_handle) {
  auto* handle = static_cast<G2PWRuntimeHandle*>(raw_handle);
  if (handle == nullptr || handle->runtime == nullptr) {
    return 0;
  }
  return handle->num_labels;
}

int g2pw_runtime_run(
    void* raw_handle,
    const std::int64_t* input_ids,
    const std::int64_t* token_type_ids,
    const std::int64_t* attention_mask,
    const float* phoneme_mask,
    const std::int64_t* char_ids,
    const std::int64_t* position_ids,
    std::int32_t batch_size,
    std::int32_t seq_len,
    float* probs) {
  auto* handle = static_cast<G2PWRuntimeHandle*>(raw_handle);
  if (handle == nullptr || handle->runtime == nullptr) {
    return static_cast<int>(g2pw::StatusCode::kInvalidArgument);
  }
  try {
    g2pw::InferenceInputs inputs{};
    inputs.input_ids = input_ids;
    inputs.token_type_ids = token_type_ids;
    inputs.attention_mask = attention_mask;
    inputs.phoneme_mask = phoneme_mask;
    inputs.char_ids = char_ids;
    inputs.position_ids = position_ids;
    inputs.batch_size = batch_size;
    inputs.seq_len = seq_len;

    g2pw::InferenceOutputs outputs{};
    outputs.probs = probs;

    const g2pw::Status status = handle->runtime->Run(inputs, outputs);
    if (!status.ok()) {
      SetError(handle, status);
      return static_cast<int>(status.code);
    }
    handle->last_error.clear();
    return static_cast<int>(g2pw::StatusCode::kOk);
  } catch (const std::exception& exc) {
    handle->last_error = exc.what();
    return static_cast<int>(g2pw::StatusCode::kInternalError);
  } catch (...) {
    handle->last_error = "unknown exception";
    return static_cast<int>(g2pw::StatusCode::kInternalError);
  }
}

}
