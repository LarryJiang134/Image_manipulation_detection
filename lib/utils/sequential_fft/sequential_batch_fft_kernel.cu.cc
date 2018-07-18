#if GOOGLE_CUDA

#include <cufft.h>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

#define CUDA_MAX_NUM_THREADS 1024
#define CUDA_MAX_NUM_BLOCKS 65535

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

inline int GET_BLOCKS(const int N) {
  int block_num = (N + CUDA_MAX_NUM_THREADS - 1) / CUDA_MAX_NUM_THREADS;
  return (block_num > CUDA_MAX_NUM_BLOCKS ? CUDA_MAX_NUM_BLOCKS : block_num);
}

__global__ void complex_inplace_scale(complex64* data, int num, float scale) {
    // one complex64 number is equivalent to two float number
    float* data_cast = reinterpret_cast<float*>(data);
    int num_cast = num * 2;
    CUDA_KERNEL_LOOP(i, num_cast) {
        data_cast[i] *= scale;
    }
}

__global__ void complex_inplace_scale(complex128* data, int num, double scale) {
    // one complex128 number is equivalent to two double number
    double* data_cast = reinterpret_cast<double*>(data);
    int num_cast = num * 2;
    CUDA_KERNEL_LOOP(i, num_cast) {
        data_cast[i] *= scale;
    }
}

inline bool cufftFunc(cufftHandle plan, const complex64 *data_in,
    complex64 *data_out, int direction) {
    // cufftExecC2C cannot take const input, hence the const_cast
    cufftComplex* in = const_cast<cufftComplex*>(
        reinterpret_cast<const cufftComplex*>(data_in));
    cufftComplex* out = reinterpret_cast<cufftComplex*>(data_out);
    return (cufftExecC2C(plan, in, out, direction) == CUFFT_SUCCESS);
}

inline bool cufftFunc(cufftHandle plan, const complex128 *data_in,
    complex128 *data_out, int direction) {
    // cufftExecZ2Z cannot take const input, hence the const_cast
    cufftDoubleComplex* in = const_cast<cufftDoubleComplex*>(
        reinterpret_cast<const cufftDoubleComplex*>(data_in));
    cufftDoubleComplex* out = reinterpret_cast<cufftDoubleComplex*>(data_out);
    return (cufftExecZ2Z(plan, in, out, direction) == CUFFT_SUCCESS);
}

bool SequentialFFTCUDAKernel(const complex64* in, int batch_size, int dim,
    int compute_size, bool forward, complex64* out) {
    bool success = true;

    // split FFT operation in sub-batches of size compute_size
    int num_compute = batch_size / compute_size;
    int stride = compute_size * dim;
    int fft_direction = (forward ? CUFFT_FORWARD : CUFFT_INVERSE);
    cufftHandle plan;
    cufftPlanMany(&plan, 1, &dim, nullptr, 0, 0, nullptr, 0, 0, CUFFT_C2C,
                  compute_size);
    for (int idx = 0; success && idx < num_compute; ++idx) {
        success = cufftFunc(plan, in + idx * stride, out + idx * stride,
                            fft_direction);
    }
    cufftDestroy(plan);

    // handle the remaining samples
    int remain_num = batch_size - num_compute * compute_size;
    if (success && remain_num > 0) {
        cufftHandle plan_remain;
        cufftPlanMany(&plan_remain, 1, &dim, nullptr, 0, 0, nullptr, 0, 0,
                      CUFFT_C2C, remain_num);
        success = cufftFunc(plan_remain, in + num_compute * stride,
                            out + num_compute * stride,
                            fft_direction);
        cufftDestroy(plan_remain);
    }

    // scale the output by 1/dim in inverse FFT (due to cuFFF implementation)
    if (success && !forward) {
        int num = batch_size * dim;
        cudaDeviceSynchronize();
        complex_inplace_scale<<<GET_BLOCKS(num*2), CUDA_MAX_NUM_THREADS>>>(out,
            num, 1. / dim);
        success = (cudaPeekAtLastError() == cudaSuccess);
    }

    if (success) { cudaDeviceSynchronize(); }  // Synchronize before returning
    return success;
}

bool SequentialFFTCUDAKernel(const complex128* in, int batch_size, int dim,
    int compute_size, bool forward, complex128* out) {
    bool success = true;

    // split FFT operation in sub-batches of size compute_size
    int num_compute = batch_size / compute_size;
    int stride = compute_size * dim;
    int fft_direction = (forward ? CUFFT_FORWARD : CUFFT_INVERSE);
    cufftHandle plan;
    cufftPlanMany(&plan, 1, &dim, nullptr, 0, 0, nullptr, 0, 0, CUFFT_Z2Z,
                  compute_size);
    for (int idx = 0; success && idx < num_compute; ++idx) {
        success = cufftFunc(plan, in + idx * stride, out + idx * stride,
                            fft_direction);
    }
    cufftDestroy(plan);

    // handle the remaining samples
    int remain_num = batch_size - num_compute * compute_size;
    if (success && remain_num > 0) {
        cufftHandle plan_remain;
        cufftPlanMany(&plan_remain, 1, &dim, nullptr, 0, 0, nullptr, 0, 0,
                      CUFFT_Z2Z, remain_num);
        success = cufftFunc(plan_remain, in + num_compute * stride,
                            out + num_compute * stride,
                            fft_direction);
        cufftDestroy(plan_remain);
    }

    // scale the output by 1/dim in inverse FFT (due to cuFFF implementation)
    if (success && !forward) {
        int num = batch_size * dim;
        cudaDeviceSynchronize();
        complex_inplace_scale<<<GET_BLOCKS(num*2), CUDA_MAX_NUM_THREADS>>>(out,
            num, 1. / dim);
        success = (cudaPeekAtLastError() == cudaSuccess);
    }

    if (success) { cudaDeviceSynchronize(); }  // Synchronize before returning
    return success;
}

#endif
