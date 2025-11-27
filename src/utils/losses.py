import torch
import torch.nn.functional as F

def d_loss_hinge_acgan(d_out_real, d_out_fake, cls_real, cls_fake, labels):
    # Adversarial Loss (Hinge)
    loss_real = torch.mean(F.relu(1.0 - d_out_real))
    loss_fake = torch.mean(F.relu(1.0 + d_out_fake))
    adv_loss = loss_real + loss_fake
    
    # Class Loss (Real 이미지에 대해서만 분류 학습 권장)
    cls_loss = F.cross_entropy(cls_real, labels)
    
    return adv_loss + cls_loss

def g_loss_hinge_acgan(d_out_fake, cls_fake, labels):
    # Adversarial Loss
    adv_loss = -torch.mean(d_out_fake)
    
    # Class Loss (Generator가 해당 클래스로 분류되도록 속여야 함)
    cls_loss = F.cross_entropy(cls_fake, labels)
    
    return adv_loss + cls_loss