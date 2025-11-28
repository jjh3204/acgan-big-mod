import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# config 파일에서 설정을 가져옵니다.
from config import cfg

def get_dataloader():
    
    # 1. 이미지 전처리 정의
    # GAN은 보통 Generator의 마지막이 Tanh이므로 이미지를 -1 ~ 1로 정규화합니다.
    transform = transforms.Compose([
        transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),  # 128x128로 리사이즈
        transforms.RandomHorizontalFlip(),                # 데이터 증강 (좌우 반전)
        transforms.ToTensor(),                            # 0~1 범위의 Tensor로 변환
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # (0~1) -> (-1~1) 정규화
    ])

    # 2. 데이터셋 로드 (ImageFolder 사용)
    # root 경로 아래에 있는 폴더명(apple, banana 등)을 클래스로 자동 인식합니다.
    if not os.path.exists(cfg.DATA_PATH):
        raise FileNotFoundError(f"데이터 경로를 찾을 수 없습니다: {cfg.DATA_PATH}")

    dataset = datasets.ImageFolder(root=cfg.DATA_PATH, transform=transform)

    # 3. DataLoader 생성
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,       # 학습 시 순서를 섞음
        num_workers=cfg.NUM_WORKERS,      # 데이터 로딩 병렬 처리 (OS에 따라 조절 가능)
        drop_last=True,     # 배치가 딱 떨어지지 않을 때 마지막 자투리 버림 (GAN 학습 안정성 위함)
        pin_memory=True     # GPU 학습 시 속도 향상
    )

    return dataloader, dataset.classes
