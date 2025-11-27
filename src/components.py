import torch.nn as nn
import utils.ops as ops
from config import cfg

class ModulesMap:
    def __init__(self):
        # Config에 따라 Spectral Norm 적용 여부 결정 (YAML: True)
        self.apply_g_sn = True 
        self.apply_d_sn = True

    def g_conv2d(self, *args, **kwargs):
        return ops.snconv2d(*args, **kwargs) if self.apply_g_sn else ops.conv2d(*args, **kwargs)

    def d_conv2d(self, *args, **kwargs):
        return ops.snconv2d(*args, **kwargs) if self.apply_d_sn else ops.conv2d(*args, **kwargs)

    def g_linear(self, *args, **kwargs):
        return ops.snlinear(*args, **kwargs) if self.apply_g_sn else ops.linear(*args, **kwargs)

    def d_linear(self, *args, **kwargs):
        return ops.snlinear(*args, **kwargs) if self.apply_d_sn else ops.linear(*args, **kwargs)

    def d_embedding(self, *args, **kwargs):
        return ops.sn_embedding(*args, **kwargs) if self.apply_d_sn else ops.embedding(*args, **kwargs)

    def g_bn(self, *args, **kwargs):
        # ops.py의 ConditionalBatchNorm2d 사용
        return ops.ConditionalBatchNorm2d(*args, **kwargs)

    def d_bn(self, *args, **kwargs):
        return ops.batchnorm_2d(*args, **kwargs)

    def g_act_fn(self, x):
        return nn.ReLU(inplace=True)(x)

    def d_act_fn(self, x):
        return nn.ReLU(inplace=True)(x)

# 싱글톤 객체 생성
MODULES = ModulesMap()