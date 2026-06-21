import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(1, channels, 1, 1)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(1, channels, 1, 1)))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        std = x.var(1, keepdim=True, unbiased=False).add(self.eps).sqrt()
        return (x - mean) / std * self.weight + self.bias

class MDTA(nn.Module):
    """
    Multi-Dconv Head Transposed Attention 
    Computes attention across channels rather than pixels for efficiency.
    """
    def __init__(self, dim, num_heads, bias=False):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, return_attn=False):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, c // self.num_heads, h * w)
        k = k.reshape(b, self.num_heads, c // self.num_heads, h * w)
        v = v.reshape(b, self.num_heads, c // self.num_heads, h * w)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        # Transposed Attention: (C/head x HW) @ (HW x C/head) -> (C/head x C/head)
        
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1) # This is the part that I can use for common space between Teacher and Student

        out = (attn @ v)
        out = out.reshape(b, c, h, w)

        out = self.project_out(out)
        
        if return_attn:
            return out, attn # Return the [B, Heads, C/head, C/head] map
        return out

class GDFN(nn.Module):
    """
    Gated-Dconv Feed-Forward Network 
    Uses gating mechanism and depth-wise convolutions for local context.
    """
    def __init__(self, dim, mlp_ratio=2.66, bias=False):
        super(GDFN, self).__init__()
        hidden_features = int(dim * mlp_ratio)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class RestormerBlock(nn.Module):
    """
    A drop-in replacement block for SwinTransformerBlock.
    Note: 'window_size' and 'shift_size' are ignored as Restormer is window-free. 
    """
    def __init__(self, dim, num_heads, mlp_ratio=2.66, bias=False, **kwargs):
        super(RestormerBlock, self).__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = MDTA(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim)
        self.ffn = GDFN(dim, mlp_ratio, bias)

    def forward(self, x, return_attn=False):
        
        # This part is for attention
        if return_attn:
            attn_out, attn_map = self.attn(self.norm1(x), return_attn=True)
            x = x + attn_out
            x = x + self.ffn(self.norm2(x))
            return x, attn_map
        
        
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x