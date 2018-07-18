#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/common_shape_fns.h"

using namespace tensorflow;

REGISTER_OP("SequentialBatchFFT")
    .Attr("T: {complex64, complex128} = DT_COMPLEX64")
    .Attr("compute_size: int = 128")
    .Input("input: T")
    .Output("output: T")
    .SetShapeFn(::tensorflow::shape_inference::UnchangedShape);

REGISTER_OP("SequentialBatchIFFT")
    .Attr("T: {complex64, complex128} = DT_COMPLEX64")
    .Attr("compute_size: int = 128")
    .Input("input: T")
    .Output("output: T")
    .SetShapeFn(::tensorflow::shape_inference::UnchangedShape);

bool SequentialFFTCUDAKernel(const complex64* in, int batch_size, int dim,
    int compute_size, bool forward, complex64* out);
bool SequentialFFTCUDAKernel(const complex128* in, int batch_size, int dim,
    int compute_size, bool forward, complex128* out);

template <typename T, bool Forward>
class SequentialBatchFFTGPU : public OpKernel {
 public:
  explicit SequentialBatchFFTGPU(OpKernelConstruction* context) :
        OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("compute_size", &compute_size_));
    OP_REQUIRES(context, compute_size_ >= 1,
                errors::InvalidArgument("Need compute_size >= 1, got ",
                                        compute_size_));
  }

  void Compute(OpKernelContext* context) override {
    auto* stream = context->op_device_context()->stream();
    OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));

    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    // input should have 2 dimensions.
    OP_REQUIRES(context, input_tensor.dims() == 2,
                errors::InvalidArgument("input must be 2-dimensional"));
    // input size
    const TensorShape& shape = input_tensor.shape();
    int batch_size = shape.dim_size(0);
    int dim = shape.dim_size(1);
    auto input_data = input_tensor.flat<T>().data();

    // Create an output tensor
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, shape, &output_tensor));
    auto output_data = output_tensor->flat<T>().data();

    // compute_size should be no more than batch_size
    int actual_compute_size =
        (compute_size_ > batch_size ? batch_size : compute_size_);
    OP_REQUIRES(
          context, SequentialFFTCUDAKernel(input_data, batch_size, dim,
                                           actual_compute_size, Forward,
                                           output_data),
          errors::Internal("cuFFT kernel execution failed : input.shape=",
                           shape.DebugString()));
  }
 private:
  int compute_size_;
};

REGISTER_KERNEL_BUILDER(Name("SequentialBatchFFT").Device(DEVICE_GPU)
                        .TypeConstraint<complex64>("T"),
                        SequentialBatchFFTGPU<complex64, true>);
REGISTER_KERNEL_BUILDER(Name("SequentialBatchIFFT").Device(DEVICE_GPU)
                        .TypeConstraint<complex64>("T"),
                        SequentialBatchFFTGPU<complex64, false>);
REGISTER_KERNEL_BUILDER(Name("SequentialBatchFFT").Device(DEVICE_GPU)
                        .TypeConstraint<complex128>("T"),
                        SequentialBatchFFTGPU<complex128, true>);
REGISTER_KERNEL_BUILDER(Name("SequentialBatchIFFT").Device(DEVICE_GPU)
                        .TypeConstraint<complex128>("T"),
                        SequentialBatchFFTGPU<complex128, false>);
