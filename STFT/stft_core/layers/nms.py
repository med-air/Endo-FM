# Copyright (c) SenseTime Research and its affiliates. All Rights Reserved.
# from ._utils import _C
from stft_core import _C

# from apex import amp

# Only valid with fp32 inputs - give AMP the hint
# nms = amp.float_function(_C.nms)
nms = _C.nms

# nms.__doc__ = """
# This function performs Non-maximum suppresion"""
