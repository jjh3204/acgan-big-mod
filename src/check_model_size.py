import torch
import os
import sys

# 경로 설정 (필요시)
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
from config import cfg
from big_resnet import Generator
from components import MODULES

# (ModelConfigStub 클래스는 기존 export.py에 있는 것 사용)
class ModelConfigStub:
    def __init__(self):
        self.info_type = "N/A"
        self.g_info_injection = "cBN"
        self.backbone = "big_resnet"
        self.z_dim = cfg.Z_DIM
        self.z_prior = "gaussian"

def measure_model_size(checkpoint_path):
    print(f"Checking model size for: {checkpoint_path}")

    # 1. 파일 자체의 크기 (체크포인트)
    if os.path.exists(checkpoint_path):
        file_size_mb = os.path.getsize(checkpoint_path) / 1024 / 1024
        print(f"📦 [CheckPoint] 전체 파일 크기: {file_size_mb:.2f} MB (옵티마이저 포함 가능성 있음)")
    
    # 2. 모델 로드 후 순수 파라미터 크기 측정
    model_cfg = ModelConfigStub()
    G = Generator(
        z_dim=cfg.Z_DIM, g_shared_dim=cfg.G_SHARED_DIM, img_size=cfg.IMG_SIZE,
        g_conv_dim=cfg.G_CONV_DIM, apply_attn=cfg.APPLY_ATTN, attn_g_loc=cfg.ATTN_G_LOC,
        g_cond_mtd="cBN", num_classes=cfg.NUM_CLASSES,
        g_init='ortho', g_depth=None, mixed_precision=False,
        MODULES=MODULES, MODEL=model_cfg
    ).cpu()

    # 가중치만 로드
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model' in checkpoint: state_dict = checkpoint['model']
    else: state_dict = checkpoint
    
    # 키 정리 및 로드
    new_state_dict = {k.replace('module.', '').replace('backbone.', ''): v for k, v in state_dict.items()}
    G.load_state_dict(new_state_dict, strict=False)
    
    # [핵심] 순수 파라미터 개수 및 용량 계산
    param_size = 0
    param_count = 0
    for param in G.parameters():
        param_count += param.numel()
        param_size += param.numel() * param.element_size() # element_size: float32는 4바이트
        
    buffer_size = 0
    for buffer in G.buffers():
        buffer_size += buffer.numel() * buffer.element_size()

    total_size_mb = (param_size + buffer_size) / 1024 / 1024

    print("-" * 40)
    print(f"🔢 총 파라미터 개수: {param_count:,} 개")
    print(f"💾 [Pure Model] 순수 모델 메모리 용량 (FP32): {total_size_mb:.2f} MB")
    print("-" * 40)
    print("※ 이 '순수 모델 용량'이 양자화 전의 기준 크기입니다.")
    print("※ TFLite(FP16) 변환 시 이 크기의 약 50%가 됩니다.")
    print("※ TFLite(INT8) 변환 시 이 크기의 약 25%가 됩니다.")

if __name__ == "__main__":
    # 가장 최신 체크포인트 자동 선택
    if os.path.exists(cfg.CHECKPOINT_DIR):
        ckpt_files = sorted([f for f in os.listdir(cfg.CHECKPOINT_DIR) if f.startswith('G_epoch') and f.endswith('.pth')])
        if ckpt_files:
            target_ckpt = os.path.join(cfg.CHECKPOINT_DIR, ckpt_files[-1])
            measure_model_size(target_ckpt)
        else:
            print("체크포인트 파일이 없습니다.")