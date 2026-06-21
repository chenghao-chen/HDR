import torch

ckpt = torch.load("./models_p1_moe_Teacher_MobileHDR_20260605_0848/phase1_best.pth", map_location="cpu")

print("epoch:", ckpt['epoch'])
print("phase:", ckpt['phase'])
print("mode:", ckpt['mode'])
print("loss:", ckpt['loss'])
print("best_psnr_mu:", ckpt['best_psnr_mu'])