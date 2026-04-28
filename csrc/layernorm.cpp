#include <sycl/sycl.hpp>

#include <algorithm>
#include <type_traits>
#include <ATen/DeviceGuard.h>
#include "utils.h"
#include "dispatch_utils.h"

#if defined(__SYCL_DEVICE_ONLY__)
#define VLLM_REQD_SG_32 [[sycl::reqd_sub_group_size(32)]]
#define VLLM_REQD_SG_16 [[sycl::reqd_sub_group_size(16)]]
#else
#define VLLM_REQD_SG_32
#define VLLM_REQD_SG_16
#endif

namespace vllm {

template <typename scalar_t>
struct alignas(8) vec4_t {
  scalar_t val[4];
};

// The vector width is fixed at 4 to avoid excessive branching in the kernel,
// which could degrade performance.
template <typename scalar_t, int NUM_DIMS, int VEC_SIZE = 4>
class rms_norm_kernel {
 public:
  rms_norm_kernel(
      scalar_t* out_,
      const scalar_t* input_,
      const int64_t input_stride_d2_,  // input.stride(-2)
      const int64_t input_stride_d3_,  // input.stride(-3)
      const int64_t input_stride_d4_,  // input.stride(-4)
      const int64_t input_shape_d2_,   // input.size(-2)
      const int64_t input_shape_d3_,   // input.size(-3)
      const scalar_t* weight_,
      const float epsilon_,
      const int num_tokens_,
      const int hidden_size_,
      sycl::local_accessor<float, 1> s_variance_)
      : out(out_),
        input(input_),
        input_stride_d2(input_stride_d2_),
        input_stride_d3(input_stride_d3_),
        input_stride_d4(input_stride_d4_),
        input_shape_d2(input_shape_d2_),
        input_shape_d3(input_shape_d3_),
        weight(weight_),
        epsilon(epsilon_),
        num_tokens(num_tokens_),
        hidden_size(hidden_size_),
        s_variance(s_variance_) {}

  void operator() VLLM_REQD_SG_32 (
      const sycl::nd_item<3>& item_ct1) const {
    float* s_variance_ptr =
        s_variance.template get_multi_ptr<sycl::access::decorated::no>().get();
    float variance = 0.0f;

    const scalar_t* input_row;
    if constexpr (NUM_DIMS == 2) {
      // 2D for layernorm normal case [batch_size, hidden]
      input_row = input + item_ct1.get_group(2) * input_stride_d2;
    } else if constexpr (NUM_DIMS == 3) {
      // 3D for q/k norm [batch_size, num_heads, head_size]
      int batch_idx = item_ct1.get_group(2) / input_shape_d2;
      int head_idx = item_ct1.get_group(2) % input_shape_d2;
      input_row =
          input + batch_idx * input_stride_d3 + head_idx * input_stride_d2;
    } else if constexpr (NUM_DIMS == 4) {
      // 4D for transformers model_impl qk norm [batch, seq, head, head_dim]
      int batch_idx = item_ct1.get_group(2) / (input_shape_d3 * input_shape_d2);
      int remaining = item_ct1.get_group(2) % (input_shape_d3 * input_shape_d2);
      int seq_idx = remaining / input_shape_d2;
      int head_idx = remaining % input_shape_d2;
      input_row = input + batch_idx * input_stride_d4 +
                  seq_idx * input_stride_d3 + head_idx * input_stride_d2;
    }

    auto vec_op = [&variance](
                      const vec4_t<scalar_t>& vec, int vec_size = VEC_SIZE) {
      for (int i = 0; i < vec_size; ++i) {
        float x = static_cast<float>(vec.val[i]);
        variance += x * x;
      }
    };
    auto scalar_op = [&variance](const scalar_t& val) {
      float x = static_cast<float>(val);
      variance += x * x;
    };

    constexpr int WIDTH = VEC_SIZE * sizeof(scalar_t);
    uintptr_t addr_in = reinterpret_cast<uintptr_t>(input_row);

    // fast path when the whole region is already aligned
    bool can_vec =
        ((addr_in & (WIDTH - 1)) == 0) && ((hidden_size & (VEC_SIZE - 1)) == 0);
    if (can_vec) {
      int64_t const num_vec_elems = hidden_size / VEC_SIZE;
      auto const* vec_in = reinterpret_cast<const vec4_t<scalar_t>*>(input_row);
      for (int i = item_ct1.get_local_id(2); i < num_vec_elems;
           i += item_ct1.get_local_range(2)) {
        vec4_t<scalar_t> tmp = vec_in[i];
        vec_op(tmp);
      }
    } else {
      int misalignment_offset = addr_in & (WIDTH - 1);
      int alignment_bytes = WIDTH - misalignment_offset;
      int prefix_elems = alignment_bytes & (WIDTH - 1);
      prefix_elems /= sizeof(scalar_t);
      prefix_elems = prefix_elems < hidden_size ? prefix_elems : hidden_size;

      // 1. handle the possibly unaligned prefix with scalar access.
      for (int i = item_ct1.get_local_id(2); i < prefix_elems;
           i += item_ct1.get_local_range(2)) {
        scalar_op(input_row[i]);
      }

      int64_t const num_vec_elems = (hidden_size - prefix_elems) / VEC_SIZE;
      auto const* vec_in =
          reinterpret_cast<const vec4_t<scalar_t>*>(input_row + prefix_elems);
      for (int i = item_ct1.get_local_id(2); i < num_vec_elems;
           i += item_ct1.get_local_range(2)) {
        vec4_t<scalar_t> tmp = vec_in[i];
        vec_op(tmp);
      }

      // 3. handle remaining tail elements.
      for (int i = item_ct1.get_local_id(2) + num_vec_elems * VEC_SIZE;
           i < hidden_size - prefix_elems;
           i += item_ct1.get_local_range(2)) {
        scalar_op((input_row + prefix_elems)[i]);
      }
    }

    variance = sycl::reduce_over_group(
        sycl::ext::oneapi::this_work_item::get_work_group<3>(),
        variance,
        sycl::plus<>());
    if (item_ct1.get_local_id(2) == 0) {
      *s_variance_ptr = sycl::rsqrt(variance / hidden_size + epsilon);
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    scalar_t* out_row = out + item_ct1.get_group(2) * hidden_size;
    uintptr_t addr_weight = reinterpret_cast<uintptr_t>(weight);
    uintptr_t addr_out = reinterpret_cast<uintptr_t>(out_row);
    bool can_vec_out = ((addr_in & (WIDTH - 1)) == 0) &&
                       ((addr_weight & (WIDTH - 1)) == 0) &&
                       ((addr_out & (WIDTH - 1)) == 0) &&
                       ((hidden_size & (VEC_SIZE - 1)) == 0);
    if (can_vec_out) {
      auto* v_in = reinterpret_cast<const vec4_t<scalar_t>*>(input_row);
      auto* v_w = reinterpret_cast<const vec4_t<scalar_t>*>(weight);
      auto* v_out = reinterpret_cast<vec4_t<scalar_t>*>(out_row);
      int64_t const out_num_vec_elems = hidden_size / VEC_SIZE;
      float s_variance_val = *s_variance_ptr;
      for (int idx = item_ct1.get_local_id(2); idx < out_num_vec_elems;
           idx += item_ct1.get_local_range(2)) {
        vec4_t<scalar_t> dst;
        vec4_t<scalar_t> src1 = v_in[idx];
        vec4_t<scalar_t> src2 = v_w[idx];
        for (int j = 0; j < VEC_SIZE; j++) {
          float x = static_cast<float>(src1.val[j]);
          dst.val[j] = ((scalar_t)(x * s_variance_val)) * src2.val[j];
        }
        v_out[idx] = dst;
      }
    } else {
      for (int idx = item_ct1.get_local_id(2); idx < hidden_size;
           idx += item_ct1.get_local_range(2)) {
        float x = (float)input_row[idx];
        out_row[idx] = ((scalar_t)(x * (*s_variance_ptr))) * weight[idx];
      }
    }
  }

 private:
  scalar_t* __restrict__ out;          // [..., hidden_size]
  const scalar_t* __restrict__ input;  // [..., hidden_size]
  const int64_t input_stride_d2;
  const int64_t input_stride_d3;
  const int64_t input_stride_d4;
  const int64_t input_shape_d2;
  const int64_t input_shape_d3;
  const scalar_t* __restrict__ weight;  // [hidden_size]
  const float epsilon;
  const int num_tokens;
  const int hidden_size;
  sycl::local_accessor<float, 1> s_variance;
};

template <typename scalar_t>
void call_rms_norm_kernel(
    torch::Tensor& out,
    torch::Tensor& input,
    torch::Tensor& weight,
    float epsilon) {
  using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;
  int num_dims = input.dim();
  int64_t input_stride_d2 = input.stride(-2);
  int64_t input_stride_d3 = (num_dims >= 3) ? input.stride(-3) : 0;
  int64_t input_stride_d4 = (num_dims >= 4) ? input.stride(-4) : 0;
  int64_t input_shape_d2 = (num_dims >= 3) ? input.size(-2) : 0;
  int64_t input_shape_d3 = (num_dims >= 4) ? input.size(-3) : 0;

  auto out_ptr = out.data_ptr<scalar_t>();
  auto input_ptr = input.data_ptr<scalar_t>();
  auto weight_ptr = weight.data_ptr<scalar_t>();
  sycl::range<3> grid(1, 1, num_tokens);
  sycl::range<3> block(1, 1, std::min(hidden_size, 1024));
  auto& queue = c10::xpu::getCurrentXPUStream().queue();

  VLLM_DISPATCH_RANK234(num_dims, [&]() {
    queue.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<float, 1> s_variance(sycl::range<1>(1), cgh);
      cgh.parallel_for(
          sycl::nd_range<3>(grid * block, block),
          vllm::rms_norm_kernel<sycl_t, tensor_rank>(
              (sycl_t*)out_ptr,
              (const sycl_t*)input_ptr,
              input_stride_d2,
              input_stride_d3,
              input_stride_d4,
              input_shape_d2,
              input_shape_d3,
              (const sycl_t*)weight_ptr,
              epsilon,
              num_tokens,
              hidden_size,
              s_variance));
    });
  });
}

constexpr int HEAD_RMS_NORM_SIMD = 16;

struct SyclKerConfigBase {};

#define __SYCL_KER_CONFIG_CONVENTION__ SyclKerConfigBase
#define SYCL_REQD_SUB_GROUP_SIZE(SIZE) [[sycl::reqd_sub_group_size(SIZE)]]

template <typename T>
using sycl_local_acc_t = sycl::local_accessor<T, 1>;

template <typename T, int N>
struct alignas(16) aligned_vector {
  T val[N];
};

struct HeadSelection {
  int use_head_selection;
  int64_t total_heads;
  int64_t head_dim;
  int64_t norm_head_start;
  int64_t norm_head_num;
};

inline int64_t head_row_offset(int64_t row_idx, const HeadSelection& selection) {
  if (!selection.use_head_selection) {
    return row_idx * selection.head_dim;
  }

  const int64_t token_idx = row_idx / selection.norm_head_num;
  const int64_t head_idx = row_idx - token_idx * selection.norm_head_num;
  return ((token_idx * selection.total_heads) +
          selection.norm_head_start +
          head_idx) * selection.head_dim;
}

template <typename T>
bool head_can_vectorize(const T* ptr, int alignment) {
  uint64_t addr = reinterpret_cast<uint64_t>(ptr);
  return addr % alignment == 0;
}

template <typename KernelFn>
inline void sycl_kernel_submit(
    const sycl::range<1>& global_range,
    const sycl::range<1>& local_range,
    sycl::queue& queue,
    KernelFn kfn) {
  queue.submit([&](sycl::handler& cgh) {
    kfn.sycl_ker_config_convention(cgh);
    cgh.parallel_for(sycl::nd_range<1>(global_range, local_range), kfn);
  });
}

template <typename KernelFn>
inline void sycl_kernel_submit(
    const sycl::range<2>& global_range,
    const sycl::range<2>& local_range,
    sycl::queue& queue,
    KernelFn kfn) {
  queue.submit([&](sycl::handler& cgh) {
    kfn.sycl_ker_config_convention(cgh);
    cgh.parallel_for(sycl::nd_range<2>(global_range, local_range), kfn);
  });
}

namespace head_rms_norm_impl_detail {

constexpr int granularity = 16;

inline int next_pow2(int val) {
  int result = 1;
  while (result < val) {
    result <<= 1;
  }
  return result;
}

}  // namespace head_rms_norm_impl_detail

template <typename T, int UNROLL, int threadsPerGroup, int maxThreads>
struct HeadRMSNormKernelFunctor : public __SYCL_KER_CONFIG_CONVENTION__ {
  static constexpr int T_per_load =
      head_rms_norm_impl_detail::granularity / sizeof(T);

  SYCL_REQD_SUB_GROUP_SIZE(HEAD_RMS_NORM_SIMD)
  void operator()(sycl::nd_item<2> item_id) const {
    constexpr int groups_per_block = maxThreads / threadsPerGroup;
    const int row_in_block = item_id.get_local_id(0);
    const int tid = item_id.get_local_id(1);
    const int row_idx = item_id.get_group(1) * groups_per_block + row_in_block;

    if (row_idx >= M_) {
      return;
    }

    const int thread_offset = tid * T_per_load;
    const int stride = threadsPerGroup * T_per_load;

    float var_sum = 0.f;
    T* row_data = X_ + head_row_offset(static_cast<int64_t>(row_idx), selection_);

    T local_buffer[UNROLL * T_per_load];

#pragma unroll
    for (int i = 0; i < UNROLL; i++) {
      T* iteration_buffer = local_buffer + i * T_per_load;
      const int iter_offset = i * stride + thread_offset;

      if (aligned_mode_) {
        const bool do_loads = (iter_offset < N_);
        if (do_loads) {
          using vec_t = aligned_vector<T, T_per_load>;
          *reinterpret_cast<vec_t*>(iteration_buffer) =
              *reinterpret_cast<const vec_t*>(row_data + iter_offset);
        } else {
#pragma unroll
          for (int j = 0; j < T_per_load; j++) {
            iteration_buffer[j] = T(0);
          }
        }
#pragma unroll
        for (int j = 0; j < T_per_load; j++) {
          float up_cast = static_cast<float>(iteration_buffer[j]);
          var_sum += up_cast * up_cast;
        }
      } else {
#pragma unroll
        for (int j = 0; j < T_per_load; j++) {
          const int idx = iter_offset + j;
          T v = (idx < N_) ? row_data[idx] : T(0);
          iteration_buffer[j] = v;
          float up_cast = static_cast<float>(v);
          var_sum += up_cast * up_cast;
        }
      }
    }

    auto sg = item_id.get_sub_group();

    if constexpr (threadsPerGroup <= HEAD_RMS_NORM_SIMD) {
      int sg_local_id = sg.get_local_linear_id();
      int local_tid = sg_local_id % threadsPerGroup;

#pragma unroll
      for (int offset = threadsPerGroup / 2; offset > 0; offset >>= 1) {
        float shifted = sycl::shift_group_left(sg, var_sum, offset);
        if (local_tid < offset) {
          var_sum += shifted;
        }
      }

      int partition_in_sg = sg_local_id / threadsPerGroup;
      var_sum = sycl::select_from_group(
          sg, var_sum, partition_in_sg * threadsPerGroup);
    } else {
#pragma unroll
      for (int offset = HEAD_RMS_NORM_SIMD / 2; offset > 0; offset >>= 1) {
        var_sum += sycl::shift_group_left(sg, var_sum, offset);
      }

      constexpr int num_warps = threadsPerGroup / HEAD_RMS_NORM_SIMD;
      int sg_local_id = sg.get_local_linear_id();
      int warp_id = tid / HEAD_RMS_NORM_SIMD;

      if (sg_local_id == 0) {
        shared_[warp_id] = var_sum;
      }
      sycl::group_barrier(item_id.get_group());

      if (warp_id == 0) {
        var_sum = (sg_local_id < num_warps) ? shared_[sg_local_id] : 0.f;
#pragma unroll
        for (int offset = HEAD_RMS_NORM_SIMD / 2; offset > 0; offset >>= 1) {
          var_sum += sycl::shift_group_left(sg, var_sum, offset);
        }
        if (sg_local_id == 0) {
          shared_[0] = var_sum;
        }
      }
      sycl::group_barrier(item_id.get_group());
      var_sum = shared_[0];
    }

    const float var = var_sum / static_cast<float>(N_);
    const float denom = sycl::rsqrt(var + epsilon_);

#pragma unroll
    for (int i = 0; i < UNROLL; i++) {
      T* iteration_buffer = local_buffer + i * T_per_load;
      const int iter_offset = i * stride + thread_offset;

      T gamma_local[T_per_load];
      if (aligned_mode_) {
        if (gamma_ != nullptr && iter_offset < N_) {
          using vec_t = aligned_vector<T, T_per_load>;
          *reinterpret_cast<vec_t*>(gamma_local) =
              *reinterpret_cast<const vec_t*>(gamma_ + iter_offset);
        } else {
#pragma unroll
          for (int j = 0; j < T_per_load; j++) {
            gamma_local[j] = T(0);
          }
        }
      } else {
        if (gamma_ != nullptr) {
#pragma unroll
          for (int j = 0; j < T_per_load; j++) {
            const int idx = iter_offset + j;
            gamma_local[j] = (idx < N_) ? gamma_[idx] : T(0);
          }
        }
      }

#pragma unroll
      for (int j = 0; j < T_per_load; j++) {
        float val = static_cast<float>(iteration_buffer[j]) * denom;
        if (gamma_ != nullptr) {
          val *= static_cast<float>(gamma_local[j]);
        }
        iteration_buffer[j] = static_cast<T>(val);
      }

      if (aligned_mode_) {
        if (iter_offset < N_) {
          using vec_t = aligned_vector<T, T_per_load>;
          *reinterpret_cast<vec_t*>(row_data + iter_offset) =
              *reinterpret_cast<const vec_t*>(iteration_buffer);
        }
      } else {
#pragma unroll
        for (int j = 0; j < T_per_load; j++) {
          const int idx = iter_offset + j;
          if (idx < N_) {
            row_data[idx] = iteration_buffer[j];
          }
        }
      }
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    constexpr int shared_size =
        (threadsPerGroup > HEAD_RMS_NORM_SIMD) ? (threadsPerGroup / HEAD_RMS_NORM_SIMD) : 1;
    shared_ = sycl_local_acc_t<float>(shared_size, cgh);
  }

  HeadRMSNormKernelFunctor(
      int N,
      int M,
      float epsilon,
      T* X,
      const T* gamma,
      HeadSelection selection,
      bool aligned_mode)
      : N_(N),
        M_(M),
        epsilon_(epsilon),
        X_(X),
        gamma_(gamma),
        selection_(selection),
        aligned_mode_(aligned_mode) {}

 private:
  int N_;
  int M_;
  float epsilon_;
  T* X_;
  const T* gamma_;
  HeadSelection selection_;
  bool aligned_mode_;
  sycl_local_acc_t<float> shared_;
};

template <typename T, int TPB>
struct HeadRMSNormLargeNKernelFunctor : public __SYCL_KER_CONFIG_CONVENTION__ {
  static constexpr int T_per_load =
      head_rms_norm_impl_detail::granularity / sizeof(T);
  static constexpr int NWARPS = TPB / HEAD_RMS_NORM_SIMD;

  SYCL_REQD_SUB_GROUP_SIZE(HEAD_RMS_NORM_SIMD)
  void operator()(sycl::nd_item<1> item_id) const {
    const int row = static_cast<int>(item_id.get_group(0));
    if (row >= M_) {
      return;
    }
    const int tid = static_cast<int>(item_id.get_local_id(0));

    T* row_data = X_ + head_row_offset(static_cast<int64_t>(row), selection_);

    float var_sum = 0.f;
    if (can_vec_) {
      using vec_t = aligned_vector<T, T_per_load>;
      const int n_vec = N_ / T_per_load;
      const int tail_start = n_vec * T_per_load;
      const vec_t* X_vec = reinterpret_cast<const vec_t*>(row_data);
      for (int i = tid; i < n_vec; i += TPB) {
        vec_t v = X_vec[i];
#pragma unroll
        for (int j = 0; j < T_per_load; ++j) {
          float f = static_cast<float>(v.val[j]);
          var_sum += f * f;
        }
      }
      for (int i = tail_start + tid; i < N_; i += TPB) {
        float f = static_cast<float>(row_data[i]);
        var_sum += f * f;
      }
    } else {
      for (int i = tid; i < N_; i += TPB) {
        float f = static_cast<float>(row_data[i]);
        var_sum += f * f;
      }
    }

    auto sg = item_id.get_sub_group();
#pragma unroll
    for (int off = HEAD_RMS_NORM_SIMD / 2; off > 0; off >>= 1) {
      var_sum += sycl::shift_group_left(sg, var_sum, off);
    }
    const int sg_lane = static_cast<int>(sg.get_local_linear_id());
    const int wid = tid / HEAD_RMS_NORM_SIMD;
    if (sg_lane == 0) {
      shared_[wid] = var_sum;
    }
    sycl::group_barrier(item_id.get_group());
    if (wid == 0) {
      float v = (sg_lane < NWARPS) ? shared_[sg_lane] : 0.f;
#pragma unroll
      for (int off = HEAD_RMS_NORM_SIMD / 2; off > 0; off >>= 1) {
        v += sycl::shift_group_left(sg, v, off);
      }
      if (sg_lane == 0) {
        shared_[0] = v;
      }
    }
    sycl::group_barrier(item_id.get_group());
    const float total = shared_[0];
    const float denom = sycl::rsqrt(total / static_cast<float>(N_) + epsilon_);

    if (can_vec_) {
      using vec_t = aligned_vector<T, T_per_load>;
      const int n_vec = N_ / T_per_load;
      const int tail_start = n_vec * T_per_load;
      const vec_t* G_vec = (gamma_ != nullptr)
          ? reinterpret_cast<const vec_t*>(gamma_)
          : nullptr;
      vec_t* X_vec = reinterpret_cast<vec_t*>(row_data);
      for (int i = tid; i < n_vec; i += TPB) {
        vec_t v = X_vec[i];
        vec_t g;
        if (G_vec != nullptr) {
          g = G_vec[i];
        }
        vec_t out;
#pragma unroll
        for (int j = 0; j < T_per_load; ++j) {
          float val = static_cast<float>(v.val[j]) * denom;
          if (G_vec != nullptr) {
            val *= static_cast<float>(g.val[j]);
          }
          out.val[j] = static_cast<T>(val);
        }
        X_vec[i] = out;
      }
      for (int i = tail_start + tid; i < N_; i += TPB) {
        float val = static_cast<float>(row_data[i]) * denom;
        if (gamma_ != nullptr) {
          val *= static_cast<float>(gamma_[i]);
        }
        row_data[i] = static_cast<T>(val);
      }
    } else {
      for (int i = tid; i < N_; i += TPB) {
        float val = static_cast<float>(row_data[i]) * denom;
        if (gamma_ != nullptr) {
          val *= static_cast<float>(gamma_[i]);
        }
        row_data[i] = static_cast<T>(val);
      }
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    shared_ = sycl_local_acc_t<float>(NWARPS, cgh);
  }

  HeadRMSNormLargeNKernelFunctor(
      int N,
      int M,
      float epsilon,
      T* X,
      const T* gamma,
      HeadSelection selection,
      bool can_vec)
      : N_(N),
        M_(M),
        epsilon_(epsilon),
        X_(X),
        gamma_(gamma),
        selection_(selection),
        can_vec_(can_vec) {}

 private:
  int N_;
  int M_;
  float epsilon_;
  T* X_;
  const T* gamma_;
  HeadSelection selection_;
  bool can_vec_;
  sycl_local_acc_t<float> shared_;
};

template <typename T>
void launch_head_rms_norm_large_n_kernel(
    int N,
    int M,
    float eps,
    T* X,
    const T* gamma,
    HeadSelection selection,
    sycl::queue& queue) {
  constexpr int TPB = 256;
  constexpr int T_per_load =
      head_rms_norm_impl_detail::granularity / sizeof(T);
  constexpr int alignment = T_per_load * sizeof(T);

  T* row0_data = X + head_row_offset(0, selection);
  const bool can_vec = (N % T_per_load == 0) && (N >= T_per_load) &&
      head_can_vectorize(row0_data, alignment) &&
      (gamma == nullptr || head_can_vectorize(gamma, alignment));

  using KernelClass = HeadRMSNormLargeNKernelFunctor<T, TPB>;
  KernelClass kfn(N, M, eps, X, gamma, selection, can_vec);
  sycl::range<1> local_range(static_cast<size_t>(TPB));
  sycl::range<1> global_range(static_cast<size_t>(M) * TPB);
  sycl_kernel_submit(global_range, local_range, queue, kfn);
}

#define LAUNCH_HEAD_RMS_NORM_IPEX(UNROLL_VAL, TPG, MAXT)                    \
  do {                                                                       \
    using KernelClass =                                                      \
        HeadRMSNormKernelFunctor<T, UNROLL_VAL, TPG, MAXT>;                 \
    KernelClass kfn(                                                         \
        N_int,                                                               \
        M_int,                                                               \
        eps_f,                                                               \
        X_data,                                                              \
        gamma_data,                                                          \
        selection,                                                           \
        aligned_mode);                                                       \
    sycl::range<2> local_range{                                              \
        static_cast<size_t>(groups_per_block), static_cast<size_t>(TPG)};    \
    sycl::range<2> global_range{                                             \
        static_cast<size_t>(groups_per_block),                               \
        static_cast<size_t>(groups_launch) * static_cast<size_t>(TPG)};      \
    sycl_kernel_submit(global_range, local_range, queue, kfn);               \
  } while (0)

template <typename T, typename T_ACC>
void head_rms_norm_kernel_impl(
    torch::Tensor& X,
    const at::Tensor& gamma,
    int64_t M,
    int64_t N,
    T_ACC eps,
    HeadSelection selection,
    sycl::queue& queue) {
  constexpr int T_per_load =
      head_rms_norm_impl_detail::granularity / sizeof(T);

  T* X_data = X.data_ptr<T>();
  const T* gamma_data = gamma.defined() ? gamma.const_data_ptr<T>() : nullptr;

  if (M == 0) {
    return;
  }

  constexpr int kIpexMaxN = 16384;

  if (N > kIpexMaxN) {
    launch_head_rms_norm_large_n_kernel<T>(
        static_cast<int>(N),
        static_cast<int>(M),
        static_cast<float>(eps),
        X_data,
        gamma_data,
        selection,
        queue);
    return;
  }

  const int N_int = static_cast<int>(N);
  const int M_int = static_cast<int>(M);
  const float eps_f = static_cast<float>(eps);

  constexpr int maxThreads = 256;
  constexpr int internalUnroll = sizeof(T) == 4 ? 4 : 2;

  const bool is_subblock_schedule = (N_int <= 128);
  const int h_per_step =
      is_subblock_schedule ? T_per_load : T_per_load * internalUnroll;

  const int one_step_threads =
      head_rms_norm_impl_detail::next_pow2((N_int + h_per_step - 1) / h_per_step);
  const int threads_per_group =
      (one_step_threads < maxThreads) ? one_step_threads : maxThreads;

  const int groups_per_block_max = is_subblock_schedule
      ? (maxThreads + threads_per_group - 1) / threads_per_group
      : 1;
  const int groups_per_block =
      (M_int < groups_per_block_max) ? M_int : groups_per_block_max;
  const int groups_launch = (M_int + groups_per_block - 1) / groups_per_block;

  const int elems_per_step = threads_per_group * h_per_step;
  const int external_unroll = (N_int + elems_per_step - 1) / elems_per_step;

  constexpr int kAlignBytes = head_rms_norm_impl_detail::granularity;
  T* row0_data = X_data + head_row_offset(0, selection);
  const bool aligned_mode = (N_int % T_per_load == 0) &&
      head_can_vectorize(row0_data, kAlignBytes) &&
      (gamma_data == nullptr || head_can_vectorize(gamma_data, kAlignBytes));

  if (is_subblock_schedule) {
    if (threads_per_group == 1) {
      LAUNCH_HEAD_RMS_NORM_IPEX(1, 1, maxThreads);
    } else if (threads_per_group == 2) {
      LAUNCH_HEAD_RMS_NORM_IPEX(1, 2, maxThreads);
    } else if (threads_per_group == 4) {
      LAUNCH_HEAD_RMS_NORM_IPEX(1, 4, maxThreads);
    } else if (threads_per_group == 8) {
      LAUNCH_HEAD_RMS_NORM_IPEX(1, 8, maxThreads);
    } else if (threads_per_group == 16) {
      LAUNCH_HEAD_RMS_NORM_IPEX(1, 16, maxThreads);
    } else if (threads_per_group == 32) {
      LAUNCH_HEAD_RMS_NORM_IPEX(1, 32, maxThreads);
    }
  } else if (external_unroll == 1) {
    LAUNCH_HEAD_RMS_NORM_IPEX(1 * internalUnroll, maxThreads, maxThreads);
  } else if (external_unroll == 2) {
    LAUNCH_HEAD_RMS_NORM_IPEX(2 * internalUnroll, maxThreads, maxThreads);
  } else if (external_unroll == 3) {
    LAUNCH_HEAD_RMS_NORM_IPEX(3 * internalUnroll, maxThreads, maxThreads);
  } else if (external_unroll == 4) {
    LAUNCH_HEAD_RMS_NORM_IPEX(4 * internalUnroll, maxThreads, maxThreads);
  }
}

#undef LAUNCH_HEAD_RMS_NORM_IPEX

void head_rms_norm(
    torch::Tensor& input,
    torch::Tensor& weight,
    int64_t norm_head_start,
    int64_t norm_head_num,
    double epsilon) {
  TORCH_CHECK(input.is_xpu(), "input must be XPU tensor");
  TORCH_CHECK(weight.is_xpu(), "weight must be XPU tensor");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  TORCH_CHECK(input.dim() == 3, "input must be 3D [B, H, N]");
  TORCH_CHECK(weight.dim() == 1, "weight must be 1D [N]");
  TORCH_CHECK(norm_head_start >= 0, "norm_head_start must be non-negative");
  TORCH_CHECK(norm_head_num >= 0, "norm_head_num must be non-negative");

  const torch::Tensor weight_cast =
      (weight.scalar_type() == input.scalar_type())
          ? weight
          : weight.to(input.scalar_type());

  HeadSelection selection{0, 1, 0, 0, 1};
  int64_t M = 0;
  int64_t N = 0;

  const int64_t total_heads = input.size(1);
  N = input.size(2);
  TORCH_CHECK(
      norm_head_start <= total_heads,
      "norm_head_start must be within total head count");
  const int64_t max_norm_head_num = total_heads - norm_head_start;
  const int64_t effective_norm_head_num =
      (norm_head_num < max_norm_head_num) ? norm_head_num : max_norm_head_num;

  if (effective_norm_head_num == 0) {
    return;
  }

  M = input.size(0) * effective_norm_head_num;
  selection.use_head_selection = 1;
  selection.total_heads = total_heads;
  selection.head_dim = N;
  selection.norm_head_start = norm_head_start;
  selection.norm_head_num = effective_norm_head_num;

  TORCH_CHECK(weight_cast.size(0) == N, "weight size must match head_dim of input");
  auto& queue = vllm::xpu::vllmGetQueue();

  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "head_rms_norm", [&]() {
    using acc_t = typename std::conditional<
        std::is_same<scalar_t, double>::value,
        double,
        float>::type;
    head_rms_norm_kernel_impl<scalar_t, acc_t>(
        input,
        weight_cast,
        M,
        N,
        static_cast<acc_t>(epsilon),
        selection,
        queue);
  });
}

template <typename scalar_t>
class fused_add_rms_norm_kernel {
 public:
  fused_add_rms_norm_kernel(
      scalar_t* __restrict__ input_,     // [..., hidden_size]
      scalar_t* __restrict__ residual_,  // [..., hidden_size]
      const int64_t input_stride_,
      const scalar_t* __restrict__ weight_,  // [hidden_size]
      const float epsilon_,
      const int num_tokens_,
      const int hidden_size_,
      sycl::local_accessor<float, 1> s_variance_)
      : input(input_),
        residual(residual_),
        input_stride(input_stride_),
        weight(weight_),
        epsilon(epsilon_),
        num_tokens(num_tokens_),
        hidden_size(hidden_size_),
        s_variance(s_variance_) {}

  void operator() VLLM_REQD_SG_32 (
      const sycl::nd_item<3>& item_ct1) const {
    float* s_variance_ptr =
        s_variance.template get_multi_ptr<sycl::access::decorated::no>().get();
    float variance = 0.0f;

    for (int idx = item_ct1.get_local_id(2); idx < hidden_size;
         idx += item_ct1.get_local_range(2)) {
      scalar_t z = (scalar_t)input[item_ct1.get_group(2) * input_stride + idx];
      z += residual[item_ct1.get_group(2) * hidden_size + idx];
      float x = (float)z;
      variance += x * x;
      residual[item_ct1.get_group(2) * hidden_size + idx] = z;
    }

    variance = sycl::reduce_over_group(
        sycl::ext::oneapi::this_work_item::get_work_group<3>(),
        variance,
        sycl::plus<>());
    if (item_ct1.get_local_id(2) == 0) {
      *s_variance_ptr = sycl::rsqrt(variance / hidden_size + epsilon);
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    for (int idx = item_ct1.get_local_id(2); idx < hidden_size;
         idx += item_ct1.get_local_range(2)) {
      float x = (float)residual[item_ct1.get_group(2) * hidden_size + idx];
      input[item_ct1.get_group(2) * input_stride + idx] =
          ((scalar_t)(x * (*s_variance_ptr))) * weight[idx];
    }
  }

 private:
  scalar_t* __restrict__ input;     // [..., hidden_size]
  scalar_t* __restrict__ residual;  // [..., hidden_size]
  const int64_t input_stride;
  const scalar_t* __restrict__ weight;  // [hidden_size]
  const float epsilon;
  const int num_tokens;
  const int hidden_size;
  sycl::local_accessor<float, 1> s_variance;  // local memory for variance
};

template <typename scalar_t>
void call_fused_add_rms_norm_kernel(
    torch::Tensor& input,
    torch::Tensor& residual,
    torch::Tensor& weight,
    float epsilon) {
  using sycl_t = typename vllm::xpu::SyclTypeTrait<scalar_t>::Type;
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;
  auto input_ptr = input.data_ptr<scalar_t>();
  auto residual_ptr = residual.data_ptr<scalar_t>();
  auto weight_ptr = weight.data_ptr<scalar_t>();
  int64_t input_stride = input.stride(-2);
  sycl::range<3> grid(1, 1, num_tokens);
  sycl::range<3> block(1, 1, std::min(hidden_size, 1024));
  auto& queue = vllm::xpu::vllmGetQueue();
  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> s_variance(sycl::range<1>(1), cgh);
    cgh.parallel_for(
        sycl::nd_range<3>(grid * block, block),
        fused_add_rms_norm_kernel<sycl_t>(
            (sycl_t*)input_ptr,
            (sycl_t*)residual_ptr,
            input_stride,
            (const sycl_t*)weight_ptr,
            epsilon,
            num_tokens,
            hidden_size,
            s_variance));
  });
}

}  // namespace vllm

void head_rms_norm(
  torch::Tensor& input,
  torch::Tensor& weight,
  int64_t norm_head_start,
  int64_t norm_head_num,
  double epsilon) {
  const at::DeviceGuard device_guard(input.device());
  vllm::head_rms_norm(
    input, weight, norm_head_start, norm_head_num, epsilon);
}

void rms_norm(
    torch::Tensor& out,
    torch::Tensor& input,
    torch::Tensor& weight,
    double epsilon) {
  const at::DeviceGuard device_guard(input.device());
  TORCH_CHECK(out.is_contiguous());
  if (input.stride(-1) != 1) {
    input = input.contiguous();
  }
  TORCH_CHECK(input.stride(-1) == 1);
  TORCH_CHECK(weight.is_contiguous());
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "call_rms_norm_kernel", [&] {
        vllm::call_rms_norm_kernel<scalar_t>(out, input, weight, epsilon);
      });
}

void fused_add_rms_norm(
    torch::Tensor& input,
    torch::Tensor& residual,
    torch::Tensor& weight,
    double epsilon) {
  const at::DeviceGuard device_guard(input.device());
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "call_fused_add_rms_norm_kernel", [&] {
        vllm::call_fused_add_rms_norm_kernel<scalar_t>(
            input, residual, weight, epsilon);
      });
}
