import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
from tqdm import tqdm
from torch.amp import autocast # 최신 PyTorch용

# 경로 설정
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from config import cfg
from dataset import get_dataloader
from big_resnet import Generator
from components import MODULES
import utils.ops as ops

# 저장해둔 메트릭 모듈 import
from metrics.inception_net import InceptionV3
from metrics.fid import frechet_inception_distance
from metrics.ins import calculate_kl_div

# Generator 초기화용 Stub
class ModelConfigStub:
    def __init__(self):
        self.info_type = "N/A"
        self.g_info_injection = "cBN"
        self.backbone = "big_resnet"
        self.z_dim = cfg.Z_DIM
        self.z_prior = "gaussian"

def load_generator(checkpoint_path):
    """학습된 생성자를 로드합니다."""
    print(f"Loading Generator from {checkpoint_path}...")
    
    # 1. 모델 구조 생성 (config.py 설정 따름)
    model_cfg = ModelConfigStub()
    G = Generator(
        z_dim=cfg.Z_DIM, 
        g_shared_dim=cfg.G_SHARED_DIM, 
        img_size=cfg.IMG_SIZE,
        g_conv_dim=cfg.G_CONV_DIM, 
        apply_attn=cfg.APPLY_ATTN, 
        attn_g_loc=cfg.ATTN_G_LOC,
        g_cond_mtd="cBN", 
        num_classes=cfg.NUM_CLASSES, # 학습된 모델이므로 3개 클래스
        g_init='ortho', g_depth=None, mixed_precision=False,
        MODULES=MODULES, MODEL=model_cfg
    ).to(cfg.DEVICE)

    # 2. 가중치 로드
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"체크포인트 파일이 없습니다: {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 에폭 정보가 묶인 경우 처리
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # 키 이름 정리 (module. 제거 등)
    if 'state_dict' in state_dict: state_dict = state_dict['state_dict']
    elif 'G' in state_dict: state_dict = state_dict['G']
    
    new_state_dict = {k.replace('module.', '').replace('backbone.', ''): v for k, v in state_dict.items()}
    G.load_state_dict(new_state_dict, strict=False)
    
    # 3. (중요) 학습 때 Attention을 껐거나 제거했다면 여기서도 처리 필요
    # 만약 config.py에서 APPLY_ATTN = True로 뒀는데 학습때 Identity로 바꿨다면 여기서도 바꿔야 함
    # cfg.APPLY_ATTN가 False면 이미 구조에 없으므로 상관없음.
    
    G.eval() # 평가 모드 전환
    return G

def get_features_and_probs(model, data_source, is_generator=False, num_samples=None):
    model.eval()
    features = []
    probs = [] # IS 계산을 위한 확률값 저장소
    
    if is_generator:
        print("Generating Fake Images and extracting outputs...")
        num_batches = num_samples // cfg.BATCH_SIZE
        iterator = tqdm(range(num_batches), desc="Fake Outputs")
    else:
        print("Extracting outputs from Real Images...")
        iterator = tqdm(data_source, desc="Real Outputs")

    with torch.no_grad():
        for item in iterator:
            if is_generator:
                z = torch.randn(cfg.BATCH_SIZE, cfg.Z_DIM).to(cfg.DEVICE)
                labels = torch.randint(0, cfg.NUM_CLASSES, (cfg.BATCH_SIZE,)).to(cfg.DEVICE)
                with autocast(device_type='cuda', dtype=torch.float16):
                    imgs = data_source(z, labels)
                    imgs = (imgs + 1) / 2 # (-1~1) -> (0~1)
                    imgs = torch.clamp(imgs, 0, 1)
            else:
                imgs, _ = item
                imgs = imgs.to(cfg.DEVICE)
                imgs = (imgs + 1) / 2 
                imgs = torch.clamp(imgs, 0, 1)

            # Inception 통과 (features: 2048차원, logits: 1008차원)
            feat, logits = model(imgs)
            
            # 1. FID용 Feature 저장
            features.append(feat.cpu().numpy())
            
            # 2. IS용 Probability 계산 및 저장 (Softmax 적용)
            prob = F.softmax(logits, dim=1)
            probs.append(prob.cpu()) # 메모리 절약을 위해 CPU로 이동

    features = np.concatenate(features, axis=0)
    probs = torch.cat(probs, dim=0) # Tensor로 합치기
    return features, probs

def calculate_stats(activations):
    """특징 벡터들의 평균(mu)과 공분산(sigma)을 계산합니다."""
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma

def evaluate():
    # 1. 평가할 모델 경로 (가장 최근에 저장된 모델 또는 Best 모델 지정)
    # 예: checkpoints 폴더에서 가장 마지막 파일 자동 선택
    ckpt_files = sorted([f for f in os.listdir(cfg.CHECKPOINT_DIR) if f.startswith('G_epoch')])
    if not ckpt_files:
        print("평가할 체크포인트가 없습니다.")
        return
    target_ckpt = os.path.join(cfg.CHECKPOINT_DIR, ckpt_files[-1]) # 가장 최근 파일
    # target_ckpt = "./results/checkpoints/G_epoch_100.pth" # 특정 파일 지정 가능

    # 2. Inception 모델 로드
    print("Loading InceptionV3 Model...")
    inception = InceptionV3(resize_input=True, normalize_input=True).to(cfg.DEVICE)
    inception.eval()

    # 3. 진짜 이미지 데이터 통계 계산 (mu1, sigma1)
    dataloader, _ = get_dataloader()
    real_feats, _ = get_features_and_probs(inception, dataloader, is_generator=False)
    mu1, sigma1 = calculate_stats(real_feats)

    # 4. Fake Data (FID + IS용)
    G = load_generator(target_ckpt)
    fake_feats, fake_probs = get_features_and_probs(inception, G, is_generator=True, num_samples=len(real_feats))
    mu2, sigma2 = calculate_stats(fake_feats)

    # 5. 메트릭 계산
    print("Calculating Metrics...")

    fid_score = frechet_inception_distance(mu1, sigma1, mu2, sigma2)
    is_score, is_std = calculate_kl_div(fake_probs, splits=10)

    print("=" * 30)
    print(f"Model Path: {os.path.basename(target_ckpt)}")
    print(f"FID Score : {fid_score:.4f} (Lower is Better)")
    print(f"IS Score  : {is_score:.4f} ± {is_std:.4f} (Higher is Better)")
    print("=" * 30)

if __name__ == "__main__":
    evaluate()