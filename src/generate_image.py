import torch
import os
import cv2
import numpy as np
from big_resnet import Generator
from config import cfg
from components import MODULES  # components.py에서 정의된 MODULES 객체 사용

def generate_test_images(checkpoint_path):
    # 1. 설정 및 저장 경로 준비
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f">>> Using device: {device}")

    save_dir = './results/pytorch_test_samples'
    os.makedirs(save_dir, exist_ok=True)
    print(f">>> Saving images to: {save_dir}")

    # 2. 모델 초기화
    # config.py의 설정을 그대로 사용
    print(">>> Initializing Generator...")
    gen = Generator(
        z_dim=cfg.Z_DIM,
        g_shared_dim=cfg.G_SHARED_DIM,
        img_size=cfg.IMG_SIZE,
        g_conv_dim=cfg.G_CONV_DIM,
        apply_attn=cfg.APPLY_ATTN,
        attn_g_loc=cfg.ATTN_G_LOC,
        g_cond_mtd="cBN",  # BigGAN 기본 설정
        num_classes=cfg.NUM_CLASSES,
        g_init="ortho",
        g_depth="resnet",
        mixed_precision=False,
        MODULES=MODULES,
        MODEL="BigGAN"
    ).to(device)

    # 3. 학습된 가중치 로드
    print(f">>> Loading weights from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        print("Please check the path in 'if __name__ == \"__main__\":' block.")
        return

    try:
        # map_location을 사용하여 CPU/GPU 호환성 확보
        state_dict = torch.load(checkpoint_path, map_location=device)
        
        # 'module.' 접두사 제거 (DataParallel로 저장된 경우)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "")
            new_state_dict[name] = v
            
        gen.load_state_dict(new_state_dict, strict=False)
        print(">>> Model loaded successfully.")
    except Exception as e:
        print(f"Error loading state_dict: {e}")
        return

    gen.eval()

    # 4. 이미지 생성 테스트
    # 클래스 정의 (0: 사과, 1: 바나나, 2: 오렌지 - 학습 데이터셋 순서에 따라 다를 수 있음)
    classes = ["Apple", "Banana", "Orange"]
    num_samples_per_class = 10  # 각 과일당 생성할 이미지 수
    
    # [수정] 그리드 생성을 위한 리스트 초기화
    grid_rows = []

    print(">>> Starting generation...")
    
    with torch.no_grad():
        for class_idx, class_name in enumerate(classes):
            print(f"Generating {class_name}...")
            # [수정] 현재 클래스의 이미지를 담을 행 리스트 초기화
            current_class_row = []
            
            # (1) Latent Vector Z 생성 (Batch Size, Z_DIM)
            # PyTorch 학습 시 사용한 정규분포(randn) 사용
            z = torch.randn(num_samples_per_class, cfg.Z_DIM).to(device)
            
            # (2) Label 생성 (Batch Size)
            # 해당 클래스 인덱스로 가득 찬 텐서 생성
            labels = torch.full((num_samples_per_class,), class_idx, dtype=torch.long).to(device)
            
            # (3) 추론 (Inference)
            # shared_label은 None으로 두면 내부에서 labels를 이용해 임베딩을 찾음
            fake_images = gen(z, labels, eval=True)

            # (4) 후처리 및 수집 (저장 X)
            for i in range(num_samples_per_class):
                # Tensor (-1 ~ 1) -> Numpy (0 ~ 255)
                img_tensor = fake_images[i].cpu().detach()
                
                # (C, H, W) -> (H, W, C)
                img_np = img_tensor.permute(1, 2, 0).numpy()
                
                img_np = ((img_np + 1) / 2.0 * 255.0)
                
                img_np = np.clip(img_np, 0, 255).astype(np.uint8)
                
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                
                current_class_row.append(img_bgr)
            
            row_image = cv2.hconcat(current_class_row)
            grid_rows.append(row_image)

    final_grid_image = cv2.vconcat(grid_rows)

    grid_filename = "fruit_grid_samples.png"
    grid_save_path = os.path.join(save_dir, grid_filename)
    cv2.imwrite(grid_save_path, final_grid_image)
    
    print(f">>> Done! Saved grid image to: {grid_save_path}")

if __name__ == "__main__":
    # ==========================================
    # [설정] 여기서 원하는 가중치 파일 경로를 수정하세요.
    # ==========================================
    target_checkpoint_path = './results/checkpoints2/G_epoch_40.pth'
    
    generate_test_images(target_checkpoint_path)