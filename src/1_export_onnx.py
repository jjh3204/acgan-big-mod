import torch
import torch.nn as nn
import os
from big_resnet import Generator
from config import cfg
import utils.ops as ops
import components

def remove_spectral_norm_recursive(module):
    """
    모델의 모든 레이어를 순회하며 Spectral Normalization(SN) hook을 제거하고,
    정규화된 가중치를 영구적으로 고정(bake)합니다.
    TFLite 변환 시 필수적인 과정입니다.
    """
    for name, child in module.named_children():
        # 하위 모듈에 대해 재귀 호출
        remove_spectral_norm_recursive(child)
        
        # SN이 적용된 레이어인지 확인 (weight_orig가 존재하면 SN이 적용된 것임)
        if hasattr(child, 'weight_orig'):
            try:
                print(f"Removing Spectral Norm from: {name} ({type(child).__name__})")
                torch.nn.utils.remove_spectral_norm(child)
            except Exception as e:
                print(f"Skipped {name}: {e}")

def export_to_onnx(checkpoint_path):
    # 1. 모델 초기화 (기존 코드 활용)
    # MODULES 객체 가져오기
    from components import MODULES 
    
    print(">>> Loading Generator...")
    gen = Generator(
        z_dim=cfg.Z_DIM,
        g_shared_dim=cfg.G_SHARED_DIM,
        img_size=cfg.IMG_SIZE,
        g_conv_dim=cfg.G_CONV_DIM,
        apply_attn=cfg.APPLY_ATTN,
        attn_g_loc=cfg.ATTN_G_LOC,
        g_cond_mtd="cBN", # big_resnet.py의 로직에 따라 설정 (확인 필요, 보통 cBN)
        num_classes=cfg.NUM_CLASSES,
        g_init="ortho",
        g_depth="resnet", # 기본값
        mixed_precision=False,
        MODULES=MODULES,
        MODEL="BigGAN" # 기본값
    )

    # 2. 사용자 지정 체크포인트 로드
    print(f">>> Loading Checkpoint from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return
    
    try:
        # map_location='cpu'로 설정하여 GPU 없이도 변환 가능하게 함
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        
        # DataParallel 등으로 저장된 경우 'module.' 접두사 제거
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "")
            new_state_dict[name] = v
            
        gen.load_state_dict(new_state_dict, strict=False)
        print(">>> Checkpoint loaded successfully.")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    gen.eval() # 평가 모드 전환

    # 3. Spectral Normalization 제거 (경량화 및 호환성 핵심)
    print(">>> Removing Spectral Normalization hooks...")
    remove_spectral_norm_recursive(gen)

    # 4. 더미 입력 생성 (Batch Size: 1)
    dummy_z = torch.randn(1, cfg.Z_DIM)
    # Label은 LongTensor여야 함 (예: 클래스 0)
    dummy_label = torch.tensor([0], dtype=torch.long)

    # 5. ONNX Export
    output_onnx_path = "./deploy/fruit_acgan_big.onnx"
    print(f">>> Exporting to {output_onnx_path}...")
    
    torch.onnx.export(
        gen,
        (dummy_z, dummy_label), # forward의 인자 순서대로 (z, label)
        output_onnx_path,
        export_params=True,
        opset_version=11, # TFLite 호환성이 좋은 버전
        do_constant_folding=True,
        input_names=['input_z', 'input_label'],
        output_names=['output_image']
    )
    print(">>> ONNX Export Complete!")

if __name__ == "__main__":
    target_checkpoint_path = './results/checkpoints2/G_epoch_40.pth'
    export_to_onnx(target_checkpoint_path)