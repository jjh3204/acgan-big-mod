import torch
import torch.optim as optim
import torch.nn as nn
import os
import sys
from tqdm import tqdm
import time
from torch.amp import GradScaler, autocast

# utils 경로 추가 (ops.py, misc.py가 src/utils에 있으므로)
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from config import cfg
from dataset import get_dataloader
from utils.losses import d_loss_hinge_acgan, g_loss_hinge_acgan
from components import MODULES
import utils.ops as ops
from big_resnet import Generator, Discriminator
from torchvision.utils import save_image
from utils.diff_augment import apply_diffaug

# Dummy Model Config for Initialization
class ModelConfigStub:
    def __init__(self):
        self.info_type = "N/A"
        self.g_info_injection = "cBN" # YAML: g_cond_mtd="cBN"
        self.backbone = "big_resnet"
        self.z_dim = cfg.Z_DIM
        self.z_prior = "gaussian"

def load_weight_safe(model, path):
    if os.path.exists(path):
        print(f"Loading weights from {path}...")
        # map_location을 사용하여 CPU/GPU 호환성 확보
        state_dict = torch.load(path, map_location='cpu', weights_only=False)
        
        # 1. StudioGAN 체크포인트의 다양한 키 구조 대응
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'G' in state_dict:
            state_dict = state_dict['G']
        elif 'D' in state_dict:
            state_dict = state_dict['D']
            
        # 2. DataParallel 등으로 인한 접두사('module.') 제거
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # 3. [추가됨] 일부 StudioGAN 모델의 'backbone.' 접두사 제거 (misc.py 참고)
        state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items()}
        
        # 4. 가중치 로드 (strict=False로 형태가 안 맞는 레이어는 무시)
        model.load_state_dict(state_dict, strict=False)
        print("Weights loaded successfully.")
        return True
    else:
        print(f"File not found: {path}")
        return False

# [추가됨] 레이어 동결/해제 유틸리티 함수
def toggle_grad(model, on, freeze_layers_contain=None):
    for name, param in model.named_parameters():
        if freeze_layers_contain:
            # 특정 문자열이 포함된 레이어(예: shared, linear2)는 동결하지 않음 (항상 학습)
            if any(x in name for x in freeze_layers_contain):
                param.requires_grad = True
            else:
                param.requires_grad = on
        else:
            param.requires_grad = on

def remove_attention_layers(model):
    removed_count = 0
    for block_list in model.blocks:
        for i, layer in enumerate(block_list):
            if "SelfAttention" in layer.__class__.__name__:
                block_list[i] = nn.Identity()
                removed_count += 1
    print(f"Removed {removed_count} Attention layers from {model.__class__.__name__}.")

def train():
    # 예: RESUME_EPOCH = 50 이면 'G_epoch_50.pth'를 로드하고 51에폭부터 시작
    RESUME_EPOCH = 0
    # 1. Dataset
    dataloader, class_names = get_dataloader()
    print(f"Target Classes ({cfg.NUM_CLASSES}): {class_names}")

    # 2. Initialize Models (1000 classes for loading)
    print("Initializing models...")
    model_cfg = ModelConfigStub()
    
    init_classes = cfg.NUM_CLASSES if RESUME_EPOCH > 0 else cfg.PRETRAINED_CLASSES
    
    G = Generator(
        z_dim=cfg.Z_DIM, g_shared_dim=cfg.G_SHARED_DIM, img_size=cfg.IMG_SIZE,
        g_conv_dim=cfg.G_CONV_DIM, apply_attn=cfg.APPLY_ATTN, attn_g_loc=cfg.ATTN_G_LOC,
        g_cond_mtd="cBN", num_classes=init_classes,
        g_init='ortho', g_depth=None, mixed_precision=False,
        MODULES=MODULES, MODEL=model_cfg
    ).to(cfg.DEVICE)

    D = Discriminator(
        img_size=cfg.IMG_SIZE, d_conv_dim=cfg.D_CONV_DIM,
        apply_d_sn=True, apply_attn=cfg.APPLY_ATTN, attn_d_loc=cfg.ATTN_D_LOC,
        d_cond_mtd="AC", aux_cls_type="N/A", d_embed_dim=None, normalize_d_embed=False,
        num_classes=init_classes,
        d_init='ortho', d_depth=None, mixed_precision=False,
        MODULES=MODULES, MODEL=model_cfg
    ).to(cfg.DEVICE)

    # 3. Load Weights & Surgery logic
    if RESUME_EPOCH > 0:
        # [재개 모드] 저장된 내 체크포인트 로드
        print(f"Resuming training from Epoch {RESUME_EPOCH}...")
        g_path = os.path.join(cfg.CHECKPOINT_DIR, f'G_epoch_{RESUME_EPOCH}.pth')
        d_path = os.path.join(cfg.CHECKPOINT_DIR, f'D_epoch_{RESUME_EPOCH}.pth')
        
        if os.path.exists(g_path) and os.path.exists(d_path):
            G.load_state_dict(torch.load(g_path))
            D.load_state_dict(torch.load(d_path))
            print("Checkpoint loaded successfully!")
        else:
            raise FileNotFoundError(f"Checkpoints not found: {g_path} or {d_path}")
            
        # 재개 모드에서는 이미 모델 구조가 3개 클래스에 맞춰져 있으므로 Surgery 불필요
    else:
        # [전이 학습 모드] ImageNet 가중치 로드 -> 모델 수술
        load_weight_safe(G, cfg.PRETRAINED_G_PATH)
        load_weight_safe(D, cfg.PRETRAINED_D_PATH)

        print("Modifying layers for 3 classes...")
        # [G] Shared Embedding 교체
        # ops.embedding 사용
        G.shared = ops.embedding(cfg.NUM_CLASSES, cfg.G_SHARED_DIM).to(cfg.DEVICE)
        G.num_classes = cfg.NUM_CLASSES
        
        # [D] Linear Layer 교체 (ACGAN)
        # D.linear2 (aux classifier) 교체. big_resnet 구현상 d_cond_mtd="AC"일 때 linear2 사용
        in_features = D.linear2.in_features
        # ops.snlinear 사용 (Spectral Norm 적용)
        D.linear2 = ops.snlinear(in_features, cfg.NUM_CLASSES, bias=False).to(cfg.DEVICE)
        D.num_classes = cfg.NUM_CLASSES

    # 5. Optimizers
    optimizer_G = optim.Adam(G.parameters(), lr=cfg.G_LR, betas=(cfg.BETA1, cfg.BETA2))
    optimizer_D = optim.Adam(D.parameters(), lr=cfg.D_LR, betas=(cfg.BETA1, cfg.BETA2))

    # [수정됨] Warm-up 설정: 처음 5 epoch 동안은 Backbone을 얼림
    warmup_epochs = 5
    if RESUME_EPOCH < warmup_epochs:
        print(f"Freezing backbone for first {warmup_epochs} epochs...")
        # G의 shared, D의 linear2만 학습하고 나머지는 False
        toggle_grad(G, False, freeze_layers_contain=['shared'])
        toggle_grad(D, False, freeze_layers_contain=['linear2'])
    else:
        print("Resuming after warm-up period. All layers are trainable.")
        toggle_grad(G, True)
        toggle_grad(D, True)

    scaler = GradScaler('cuda')
    # 6. Training Loop
    print("Start Training...")
    
    global_step = 0

    for epoch in range(RESUME_EPOCH, cfg.EPOCHS):
        # Warm-up이 끝나면 Backbone 학습 재개
        if epoch == warmup_epochs:
            print("Unfreezing all layers! Start fine-tuning backbone.")
            toggle_grad(G, True)
            toggle_grad(D, True)

        # tqdm으로 dataloader를 감싸서 프로그레스 바 생성
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS}")

        g_loss_val = 0.0

        for real_imgs, labels in loop:
            real_imgs, labels = real_imgs.to(cfg.DEVICE), labels.to(cfg.DEVICE)
            batch_size = real_imgs.size(0)

            # --- Train D ---
            optimizer_D.zero_grad()
            z = torch.randn(batch_size, cfg.Z_DIM).to(cfg.DEVICE)
            fake_labels = torch.randint(0, cfg.NUM_CLASSES, (batch_size,)).to(cfg.DEVICE)
            
            with autocast(device_type='cuda', dtype=torch.float16):
                fake_imgs = G(z, fake_labels)
                # [⭐️핵심 수정] 판별자에 넣기 전, 둘 다 증강 적용!
                real_imgs_aug = apply_diffaug(real_imgs, cfg.DIFF_AUGMENT_POLICY)
                fake_imgs_aug = apply_diffaug(fake_imgs, cfg.DIFF_AUGMENT_POLICY)
                
                # 증강된 이미지를 D에 입력
                d_out_real = D(real_imgs_aug, labels)
                d_out_fake = D(fake_imgs_aug.detach(), fake_labels)
                '''
                d_out_real = D(real_imgs, labels)
                d_out_fake = D(fake_imgs.detach(), fake_labels)
                '''
                d_loss = d_loss_hinge_acgan(d_out_real['adv_output'], d_out_fake['adv_output'],
                                        d_out_real['cls_output'], d_out_fake['cls_output'], labels)
                
            scaler.scale(d_loss).backward()
            scaler.step(optimizer_D)
            scaler.update()
            #d_loss.backward()
            #optimizer_D.step()

            # --- Train G ---
            if global_step % cfg.D_STEPS == 0:
                optimizer_G.zero_grad()
                
                with autocast(device_type='cuda', dtype=torch.float16):
                    # [⭐️핵심 수정] G 학습 때도 증강된 이미지로 속임수를 써야 함
                    fake_imgs_g = G(z, fake_labels)
                    fake_imgs_g_aug = apply_diffaug(fake_imgs_g, cfg.DIFF_AUGMENT_POLICY)
                    d_out_gen = D(fake_imgs_g_aug, fake_labels)
                    '''
                    d_out_gen = D(fake_imgs_g, fake_labels)
                    '''
                    g_loss = g_loss_hinge_acgan(d_out_gen['adv_output'], d_out_gen['cls_output'], fake_labels)
                
                scaler.scale(g_loss).backward()
                scaler.step(optimizer_G)
                scaler.update()

                g_loss_val = g_loss.item()
            
            global_step += 1
            #g_loss.backward()
            #optimizer_G.step()

            # tqdm의 postfix로 실시간 Loss 표시
            loop.set_postfix(d_loss=f"{d_loss.item():.4f}", g_loss=f"{g_loss_val:.4f}")

        # Save Sample
        with torch.no_grad():
            sample_z = torch.randn(cfg.NUM_CLASSES * 4, cfg.Z_DIM).to(cfg.DEVICE)
            sample_y = torch.arange(cfg.NUM_CLASSES).repeat_interleave(4).to(cfg.DEVICE)
            sample_imgs = G(sample_z, sample_y)
            save_image((sample_imgs + 1) / 2, os.path.join(cfg.SAMPLE_DIR, f'epoch_{epoch+1}.png'), nrow=4)
            
            # 체크포인트 저장
            if (epoch+1) % cfg.CHECKPOINT_INTERVAL == 0:
                 torch.save(G.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, f'G_epoch_{epoch+1}.pth'))
                 torch.save(D.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, f'D_epoch_{epoch+1}.pth'))

if __name__ == "__main__":
    os.makedirs(cfg.SAMPLE_DIR, exist_ok=True)
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    train()
