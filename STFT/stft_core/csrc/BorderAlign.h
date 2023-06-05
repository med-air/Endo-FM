// Copyright (c) SenseTime Research and its affiliates. All Rights Reserved.
#pragma once

#include "cpu/vision.h"

#ifdef WITH_CUDA
#include "cuda/vision.h"
#endif

// Interface for Python
at::Tensor BorderAlign_forward(const at::Tensor& feature,
                            const at::Tensor& boxes,
                            const at::Tensor& wh,
                            const int pool_size) {
  if (feature.type().is_cuda()) {
#ifdef WITH_CUDA
    return border_align_forward_cuda(feature, boxes, wh, pool_size);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}

at::Tensor BorderAlign_backward(const at::Tensor& gradOutput,
                             const at::Tensor& feature,
                             const at::Tensor& boxes,
                             const at::Tensor& wh,
                             const int pool_size) {
  if (gradOutput.type().is_cuda()) {
#ifdef WITH_CUDA
    return border_align_backward_cuda(gradOutput, feature, boxes, wh, pool_size);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}

