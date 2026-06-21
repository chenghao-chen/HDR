import torch
import torch.nn as nn
import torch.nn.functional as F

class DifferentiableGBTF_BGGR(nn.Module):
    """
    A fully differentiable PyTorch implementation of the GBTF Demosaicing Algorithm.
    Hardcoded and optimized for BGGR layouts. Outputs true [B, 3, H, W] RGB tensors 
    ready for LPIPS VGG-16 networks.
    """
    def __init__(self):
        super().__init__()
        # 1. H/V 1D Kernels
        k_1d = torch.tensor([-0.25, 0.5, 0.5, 0.5, -0.25], dtype=torch.float32)
        self.register_buffer('HK', k_1d.view(1, 1, 1, 5))
        self.register_buffer('VK', k_1d.view(1, 1, 5, 1))

        # 2. Box filter 5x5 for Sum Sliding Windows
        self.register_buffer('box5x5', torch.ones(1, 1, 5, 5, dtype=torch.float32))

        # 3. Prb Matrix 7x7 for opposing color interpolation
        prb = torch.tensor([
            [ 0, 0, -0.0313, 0, -0.0313, 0, 0],
            [ 0, 0, 0, 0, 0, 0, 0],
            [-0.0313, 0, 0.3125, 0, 0.3125, 0, -0.0313],
            [ 0, 0, 0, 0, 0, 0, 0],
            [-0.0313, 0, 0.3125, 0, 0.3125, 0, -0.0313],
            [ 0, 0, 0, 0, 0, 0, 0],
            [ 0, 0, -0.0313, 0, -0.0313, 0, 0]
        ], dtype=torch.float32)
        self.register_buffer('Prb', prb.view(1, 1, 7, 7))

        # 4. Cross Matrix 3x3 for native green sites
        cross = torch.tensor([[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]], dtype=torch.float32)
        self.register_buffer('k_cross', cross.view(1, 1, 3, 3))

    def forward(self, mosaic):
        # mosaic is expected to be [B, 1, H, W]
        B, _, H, W = mosaic.shape
        device, dtype = mosaic.device, mosaic.dtype

        # -- Step 0: Base 3-Channel Allocation (Output natively in RGB format) --
        Dst = torch.zeros((B, 3, H, W), device=device, dtype=dtype)
        Dst[:, 2, 0::2, 0::2] = mosaic[:, 0, 0::2, 0::2] # Blue -> RGB Channel 2
        Dst[:, 1, 0::2, 1::2] = mosaic[:, 0, 0::2, 1::2] # G1   -> RGB Channel 1
        Dst[:, 1, 1::2, 0::2] = mosaic[:, 0, 1::2, 0::2] # G2   -> RGB Channel 1
        Dst[:, 0, 1::2, 1::2] = mosaic[:, 0, 1::2, 1::2] # Red  -> RGB Channel 0

        # -- Step 1: Horizontal & Vertical Difference Maps --
        pad_H = F.pad(mosaic, (2, 2, 0, 0), mode='reflect')
        HCD = F.conv2d(pad_H, self.HK)

        pad_V = F.pad(mosaic, (0, 0, 2, 2), mode='reflect')
        VCD = F.conv2d(pad_V, self.VK)

        # Vectorized checkerboard generation for BGGR
        y_idx = torch.arange(H, device=device).view(-1, 1)
        x_idx = torch.arange(W, device=device).view(1, -1)
        checker = torch.where((y_idx + x_idx) % 2 == 0, 1.0, -1.0).view(1, 1, H, W)

        HCD = checker * (HCD - mosaic)
        VCD = checker * (VCD - mosaic)

        # -- Step 2: Gradient Maps --
        HCD_pad = F.pad(HCD, (5, 5, 5, 5), mode='constant', value=0)
        VCD_pad = F.pad(VCD, (5, 5, 5, 5), mode='constant', value=0)

        HGrad = torch.abs(HCD_pad[:, :, 5:5+H, 6:6+W] - HCD_pad[:, :, 5:5+H, 4:4+W])
        VGrad = torch.abs(VCD_pad[:, :, 6:6+H, 5:5+W] - VCD_pad[:, :, 4:4+H, 5:5+W])

        # -- Step 3: Sum Sliding Windows (Box Filters) --
        pad_box_h = F.pad(HGrad, (2, 2, 2, 2), mode='constant', value=0)
        H_sum = F.conv2d(pad_box_h, self.box5x5)

        pad_box_v = F.pad(VGrad, (2, 2, 2, 2), mode='constant', value=0)
        V_sum = F.conv2d(pad_box_v, self.box5x5)

        eps = 1e-9
        # PyTorch roll identically mimics Numpy's vstack/hstack array shifting
        N_val = torch.roll(V_sum, shifts=2, dims=2)
        S_val = torch.roll(V_sum, shifts=-2, dims=2)
        E_val = torch.roll(H_sum, shifts=2, dims=3)
        W_val = torch.roll(H_sum, shifts=-2, dims=3)

        # BUG FIX: Renamed N, S, E, W to Wt_N, Wt_S, Wt_E, Wt_W so 'W' (Width) is preserved!
        Wt_N = 1.0 / (N_val**2 + eps)
        Wt_S = 1.0 / (S_val**2 + eps)
        Wt_E = 1.0 / (E_val**2 + eps)
        Wt_W = 1.0 / (W_val**2 + eps)
        TotalW = Wt_N + Wt_S + Wt_E + Wt_W

        # -- Step 4: Green Channel Interpolation --
        HCD_pad9 = F.pad(HCD, (4, 4, 0, 0), mode='reflect')
        VCD_pad9 = F.pad(VCD, (0, 0, 4, 4), mode='reflect')

        H_sum_L = HCD_pad9[:, :, :, 0:W] + HCD_pad9[:, :, :, 1:W+1] + HCD_pad9[:, :, :, 2:W+2] + HCD_pad9[:, :, :, 3:W+3]
        H_sum_R = HCD_pad9[:, :, :, 5:W+5] + HCD_pad9[:, :, :, 6:W+6] + HCD_pad9[:, :, :, 7:W+7] + HCD_pad9[:, :, :, 8:W+8]
        V_sum_U = VCD_pad9[:, :, 0:H, :] + VCD_pad9[:, :, 1:H+1, :] + VCD_pad9[:, :, 2:H+2, :] + VCD_pad9[:, :, 3:H+3, :]
        V_sum_D = VCD_pad9[:, :, 5:H+5, :] + VCD_pad9[:, :, 6:H+6, :] + VCD_pad9[:, :, 7:H+7, :] + VCD_pad9[:, :, 8:H+8, :]

        # Updated mapping for the weights
        AccumH = (Wt_E * 0.2 * H_sum_L) + ((Wt_W + Wt_E) * 0.2 * HCD) + (Wt_W * 0.2 * H_sum_R)
        AccumV = (Wt_N * 0.2 * V_sum_U) + ((Wt_N + Wt_S) * 0.2 * VCD) + (Wt_S * 0.2 * V_sum_D)

        TPdiff = (AccumH + AccumV) / (TotalW + eps)

        # BGGR masks missing green at Blue(0,0) and Red(1,1)
        mask_missing_G = ((y_idx + x_idx) % 2 == 0).view(1, 1, H, W)

        G_new = torch.clamp(mosaic + TPdiff, 0.0, 1.0)
        Dst[:, 1:2, :, :] = torch.where(mask_missing_G, G_new, mosaic)

        # -- Step 5: Red and Blue Interpolation on opposing native positions --
        TPdiff_pad = F.pad(TPdiff, (3, 3, 3, 3), mode='reflect')
        Correction = F.conv2d(TPdiff_pad, self.Prb)

        RB_val = torch.clamp(Dst[:, 1:2, :, :] - Correction, 0.0, 1.0)

        # Fully differentiable mask routing
        mask_B_native = ((y_idx % 2 == 0) & (x_idx % 2 == 0)).view(1, 1, H, W) # (0,0)
        mask_R_native = ((y_idx % 2 == 1) & (x_idx % 2 == 1)).view(1, 1, H, W) # (1,1)

        Dst[:, 2:3, :, :] = torch.where(mask_R_native, RB_val, Dst[:, 2:3, :, :]) # Interp Blue onto Red spots
        Dst[:, 0:1, :, :] = torch.where(mask_B_native, RB_val, Dst[:, 0:1, :, :]) # Interp Red onto Blue spots

        # -- Step 6: Red and Blue on native Green sites --
        GmB = Dst[:, 1:2, :, :] - Dst[:, 2:3, :, :]
        GmR = Dst[:, 1:2, :, :] - Dst[:, 0:1, :, :]

        GmB_pad = F.pad(GmB, (1, 1, 1, 1), mode='reflect')
        GmR_pad = F.pad(GmR, (1, 1, 1, 1), mode='reflect')

        GmB_smooth = F.conv2d(GmB_pad, self.k_cross)
        GmR_smooth = F.conv2d(GmR_pad, self.k_cross)

        B_rec = torch.clamp(Dst[:, 1:2, :, :] - GmB_smooth, 0.0, 1.0)
        R_rec = torch.clamp(Dst[:, 1:2, :, :] - GmR_smooth, 0.0, 1.0)

        mask_native_G = ~mask_missing_G 
        Dst[:, 2:3, :, :] = torch.where(mask_native_G, B_rec, Dst[:, 2:3, :, :])
        Dst[:, 0:1, :, :] = torch.where(mask_native_G, R_rec, Dst[:, 0:1, :, :])

        return Dst