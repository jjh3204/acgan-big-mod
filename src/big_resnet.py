# src/big_resnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.ops as ops
import utils.misc as misc

# --- GenBlock (변경 없음: 구조적 핵심) ---
class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, g_cond_mtd, affine_input_dim, MODULES):
        super(GenBlock, self).__init__()
        self.g_cond_mtd = g_cond_mtd

        self.bn1 = MODULES.g_bn(affine_input_dim, in_channels, MODULES)
        self.bn2 = MODULES.g_bn(affine_input_dim, out_channels, MODULES)

        self.activation = MODULES.g_act_fn
        self.conv2d0 = MODULES.g_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2d1 = MODULES.g_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2d2 = MODULES.g_conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, affine):
        x0 = x
        x = self.bn1(x, affine)
        x = self.activation(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv2d1(x)

        x = self.bn2(x, affine)
        x = self.activation(x)
        x = self.conv2d2(x)

        x0 = F.interpolate(x0, scale_factor=2, mode="nearest")
        x0 = self.conv2d0(x0)
        out = x + x0
        return out


# --- Generator (InfoGAN 로직 제거) ---
class Generator(nn.Module):
    def __init__(self, z_dim, g_shared_dim, img_size, g_conv_dim, apply_attn, attn_g_loc, g_cond_mtd, num_classes, g_init, g_depth,
                 mixed_precision, MODULES, MODEL):
        super(Generator, self).__init__()
        # ImageNet BigGAN 채널 설정 (구조 유지 필수)
        g_in_dims_collection = {
            "32": [g_conv_dim * 4, g_conv_dim * 4, g_conv_dim * 4],
            "64": [g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2],
            "128": [g_conv_dim * 16, g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2],
            "256": [g_conv_dim * 16, g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2],
            "512": [g_conv_dim * 16, g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2, g_conv_dim]
        }

        g_out_dims_collection = {
            "32": [g_conv_dim * 4, g_conv_dim * 4, g_conv_dim * 4],
            "64": [g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2, g_conv_dim],
            "128": [g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2, g_conv_dim],
            "256": [g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2, g_conv_dim],
            "512": [g_conv_dim * 16, g_conv_dim * 8, g_conv_dim * 8, g_conv_dim * 4, g_conv_dim * 2, g_conv_dim, g_conv_dim]
        }

        bottom_collection = {"32": 4, "64": 4, "128": 4, "256": 4, "512": 4}

        self.z_dim = z_dim
        self.g_shared_dim = g_shared_dim
        self.g_cond_mtd = g_cond_mtd
        self.num_classes = num_classes
        self.mixed_precision = mixed_precision
        self.MODEL = MODEL
        self.in_dims = g_in_dims_collection[str(img_size)]
        self.out_dims = g_out_dims_collection[str(img_size)]
        self.bottom = bottom_collection[str(img_size)]
        self.num_blocks = len(self.in_dims)
        self.chunk_size = z_dim // (self.num_blocks + 1)
        self.affine_input_dim = self.chunk_size
        
        # [삭제됨] InfoGAN 관련 로직 및 Linear Layer 제거

        self.linear0 = MODULES.g_linear(in_features=self.chunk_size, out_features=self.in_dims[0]*self.bottom*self.bottom, bias=True)

        # cBN (Conditional Batch Norm)을 위한 Shared Embedding
        if self.g_cond_mtd != "W/O":
            self.affine_input_dim += self.g_shared_dim
            self.shared = ops.embedding(num_embeddings=self.num_classes, embedding_dim=self.g_shared_dim)

        self.blocks = []
        for index in range(self.num_blocks):
            self.blocks += [[
                GenBlock(in_channels=self.in_dims[index],
                         out_channels=self.out_dims[index],
                         g_cond_mtd=self.g_cond_mtd,
                         affine_input_dim=self.affine_input_dim,
                         MODULES=MODULES)
            ]]

            if index + 1 in attn_g_loc and apply_attn:
                self.blocks += [[ops.SelfAttention(self.out_dims[index], is_generator=True, MODULES=MODULES)]]

        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        self.bn4 = ops.batchnorm_2d(in_features=self.out_dims[-1])
        self.activation = MODULES.g_act_fn
        self.conv2d5 = MODULES.g_conv2d(in_channels=self.out_dims[-1], out_channels=3, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

        ops.init_weights(self.modules, g_init)

    def forward(self, z, label, shared_label=None, eval=False):
        # [단순화] InfoGAN injection 로직 제거 (ACGAN 전용)
        affine_list = []
        with torch.cuda.amp.autocast() if self.mixed_precision and not eval else misc.dummy_context_mgr() as mp:
            zs = torch.split(z, self.chunk_size, 1)
            z = zs[0]
            
            if self.g_cond_mtd != "W/O":
                if shared_label is None:
                    shared_label = self.shared(label)
                affine_list.append(shared_label)
            
            if len(affine_list) == 0:
                affines = [item for item in zs[1:]]
            else:
                affines = [torch.cat(affine_list + [item], 1) for item in zs[1:]]

            act = self.linear0(z)
            act = act.view(-1, self.in_dims[0], self.bottom, self.bottom)
            
            counter = 0
            for index, blocklist in enumerate(self.blocks):
                for block in blocklist:
                    if isinstance(block, ops.SelfAttention):
                        act = block(act)
                    else:
                        act = block(act, affines[counter])
                        counter += 1

            act = self.bn4(act)
            act = self.activation(act)
            act = self.conv2d5(act)
            out = self.tanh(act)
        return out


# --- DiscOptBlock (변경 없음: 구조적 핵심) ---
class DiscOptBlock(nn.Module):
    def __init__(self, in_channels, out_channels, apply_d_sn, MODULES):
        super(DiscOptBlock, self).__init__()
        self.apply_d_sn = apply_d_sn

        self.conv2d0 = MODULES.d_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2d1 = MODULES.d_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2d2 = MODULES.d_conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

        if not apply_d_sn:
            self.bn0 = MODULES.d_bn(in_features=in_channels)
            self.bn1 = MODULES.d_bn(in_features=out_channels)

        self.activation = MODULES.d_act_fn
        self.average_pooling = nn.AvgPool2d(2)

    def forward(self, x):
        x0 = x
        x = self.conv2d1(x)
        if not self.apply_d_sn:
            x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2d2(x)
        x = self.average_pooling(x)

        x0 = self.average_pooling(x0)
        if not self.apply_d_sn:
            x0 = self.bn0(x0)
        x0 = self.conv2d0(x0)
        out = x + x0
        return out


# --- DiscBlock (변경 없음: 구조적 핵심) ---
class DiscBlock(nn.Module):
    def __init__(self, in_channels, out_channels, apply_d_sn, MODULES, downsample=True):
        super(DiscBlock, self).__init__()
        self.apply_d_sn = apply_d_sn
        self.downsample = downsample

        self.activation = MODULES.d_act_fn

        self.ch_mismatch = False
        if in_channels != out_channels:
            self.ch_mismatch = True

        if self.ch_mismatch or downsample:
            self.conv2d0 = MODULES.d_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
            if not apply_d_sn:
                self.bn0 = MODULES.d_bn(in_features=in_channels)

        self.conv2d1 = MODULES.d_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2d2 = MODULES.d_conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

        if not apply_d_sn:
            self.bn1 = MODULES.d_bn(in_features=in_channels)
            self.bn2 = MODULES.d_bn(in_features=out_channels)

        self.average_pooling = nn.AvgPool2d(2)

    def forward(self, x):
        x0 = x
        if not self.apply_d_sn:
            x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2d1(x)

        if not self.apply_d_sn:
            x = self.bn2(x)
        x = self.activation(x)
        x = self.conv2d2(x)
        if self.downsample:
            x = self.average_pooling(x)

        if self.downsample or self.ch_mismatch:
            if not self.apply_d_sn:
                x0 = self.bn0(x0)
            x0 = self.conv2d0(x0)
            if self.downsample:
                x0 = self.average_pooling(x0)
        out = x + x0
        return out


# --- Discriminator (ACGAN 전용으로 대폭 축소) ---
class Discriminator(nn.Module):
    def __init__(self, img_size, d_conv_dim, apply_d_sn, apply_attn, attn_d_loc, d_cond_mtd, aux_cls_type, d_embed_dim, normalize_d_embed,
                 num_classes, d_init, d_depth, mixed_precision, MODULES, MODEL):
        super(Discriminator, self).__init__()
        d_in_dims_collection = {
            "32": [3] + [d_conv_dim * 2, d_conv_dim * 2, d_conv_dim * 2],
            "64": [3] + [d_conv_dim, d_conv_dim * 2, d_conv_dim * 4, d_conv_dim * 8],
            "128": [3] + [d_conv_dim, d_conv_dim * 2, d_conv_dim * 4, d_conv_dim * 8, d_conv_dim * 16],
            "256": [3] + [d_conv_dim, d_conv_dim * 2, d_conv_dim * 4, d_conv_dim * 8, d_conv_dim * 8, d_conv_dim * 16],
            "512": [3] + [d_conv_dim, d_conv_dim, d_conv_dim * 2, d_conv_dim * 4, d_conv_dim * 8, d_conv_dim * 8, d_conv_dim * 16]
        }

        d_out_dims_collection = {
            "32": [d_conv_dim * 2, d_conv_dim * 2, d_conv_dim * 2, d_conv_dim * 2],
            "64": [d_conv_dim, d_conv_dim * 2, d_conv_dim * 4, d_conv_dim * 8, d_conv_dim * 16],
            "128": [d_conv_dim, d_conv_dim * 2, d_conv_dim * 4, d_conv_dim * 8, d_conv_dim * 16, d_conv_dim * 16],
            "256": [d_conv_dim, d_conv_dim * 2, d_conv_dim * 4, d_conv_dim * 8, d_conv_dim * 8, d_conv_dim * 16, d_conv_dim * 16],
            "512": [d_conv_dim, d_conv_dim, d_conv_dim * 2, d_conv_dim * 4, d_conv_dim * 8, d_conv_dim * 8, d_conv_dim * 16, d_conv_dim * 16]
        }

        d_down = {
            "32": [True, True, False, False],
            "64": [True, True, True, True, False],
            "128": [True, True, True, True, True, False],
            "256": [True, True, True, True, True, True, False],
            "512": [True, True, True, True, True, True, True, False]
        }

        self.d_cond_mtd = d_cond_mtd
        self.aux_cls_type = aux_cls_type
        self.normalize_d_embed = normalize_d_embed
        self.num_classes = num_classes
        self.mixed_precision = mixed_precision
        self.in_dims = d_in_dims_collection[str(img_size)]
        self.out_dims = d_out_dims_collection[str(img_size)]
        self.MODEL = MODEL
        down = d_down[str(img_size)]

        self.blocks = []
        for index in range(len(self.in_dims)):
            if index == 0:
                self.blocks += [[
                    DiscOptBlock(in_channels=self.in_dims[index], out_channels=self.out_dims[index], apply_d_sn=apply_d_sn, MODULES=MODULES)
                ]]
            else:
                self.blocks += [[
                    DiscBlock(in_channels=self.in_dims[index],
                              out_channels=self.out_dims[index],
                              apply_d_sn=apply_d_sn,
                              MODULES=MODULES,
                              downsample=down[index])
                ]]

            if index + 1 in attn_d_loc and apply_attn:
                self.blocks += [[ops.SelfAttention(self.out_dims[index], is_generator=False, MODULES=MODULES)]]

        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        self.activation = MODULES.d_act_fn

        # [변경] Linear1: Adversarial Output (Real/Fake 판별)
        # ACGAN에서는 무조건 1개의 스칼라 값만 출력
        self.linear1 = MODULES.d_linear(in_features=self.out_dims[-1], out_features=1, bias=True)

        # [변경] Linear2: Auxiliary Classifier (클래스 분류)
        # ACGAN (d_cond_mtd="AC") 전용 로직만 유지. PD, 2C 등 제거.
        if self.d_cond_mtd == "AC":
            self.linear2 = MODULES.d_linear(in_features=self.out_dims[-1], out_features=num_classes, bias=False)
        
        # [삭제됨] InfoGAN Q-Head (discrete/continuous linear layers) 제거
        # [삭제됨] TAC, ADC 관련 linear_mi, embedding_mi 제거

        if d_init:
            ops.init_weights(self.modules, d_init)

    def forward(self, x, label, eval=False, adc_fake=False):
        with torch.cuda.amp.autocast() if self.mixed_precision and not eval else misc.dummy_context_mgr() as mp:
            embed, proxy, cls_output = None, None, None
            
            h = x
            for index, blocklist in enumerate(self.blocks):
                for block in blocklist:
                    h = block(h)
            
            # Global Average Pooling
            h = self.activation(h)
            h = torch.sum(h, dim=[2, 3])

            # 1. Adversarial Output (Real/Fake)
            adv_output = torch.squeeze(self.linear1(h))

            # 2. Class Conditioning (ACGAN)
            # PD, MH, MD 등 다른 방식은 모두 제거하고 AC만 유지
            if self.d_cond_mtd == "AC":
                if self.normalize_d_embed:
                    for W in self.linear2.parameters():
                        W = F.normalize(W, dim=1)
                    h = F.normalize(h, dim=1)
                cls_output = self.linear2(h)
            
            # [삭제됨] TAC, ADC, InfoGAN 관련 Forward 로직 제거

        # ACGAN Loss 계산에 필요한 값만 리턴
        return {
            "h": h,
            "adv_output": adv_output,
            "cls_output": cls_output,
            "label": label,
        }