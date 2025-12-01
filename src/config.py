import torch

class Config:
    # 데이터 경로
    DATA_PATH = './data/fruit_dataset/train'
    
    PRETRAINED_G_PATH = './results/checkpoints/model=G-best-weights-step=200000.pth'
    PRETRAINED_D_PATH = './results/checkpoints/model=D-best-weights-step=200000.pth'

    APPLY_ATTN = True          # False로 설정하여 어텐션 끄기
    ATTN_G_LOC = [4]
    ATTN_D_LOC = [1]

    # Model Settings (YAML 기반)
    IMG_SIZE = 128
    PRETRAINED_CLASSES = 1000
    NUM_CLASSES = 3
    
    # Model Specs
    Z_DIM = 120
    G_SHARED_DIM = 128
    G_CONV_DIM = 96
    D_CONV_DIM = 96
    
    # Hyperparameters
    BATCH_SIZE = 32
    G_LR = 0.00005
    D_LR = 0.0002
    BETA1 = 0.0
    BETA2 = 0.999

    EPOCHS = 50               # 총 학습 에폭 수
    NUM_WORKERS = 4            # DataLoader의 num_workers (CPU 코어 수에 맞게 조절)
    D_STEPS = 2                # Generator 1회 업데이트 당 Discriminator 업데이트 횟수

    SAMPLE_INTERVAL = 1        # 샘플 이미지 저장 주기 (매 에폭마다)
    CHECKPOINT_INTERVAL = 10   # 모델 가중치 저장 주기 (10 에폭마다)
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    SAMPLE_DIR = './results/samples'
    CHECKPOINT_DIR = './results/checkpoints'

    DIFF_AUGMENT_POLICY = "color,translation,cutout"

cfg = Config()
