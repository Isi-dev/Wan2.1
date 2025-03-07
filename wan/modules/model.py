# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
import os
from safetensors import safe_open
import torch
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .attention import flash_attention

__all__ = ['WanModel']


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x

# def sinusoidal_embedding_1d(dim, position):
#     assert dim % 2 == 0
#     half = dim // 2
#     position = position.to(dtype=torch.float32)  # Reduced precision

#     sinusoid = torch.outer(
#         position, torch.pow(10000, -torch.arange(half, dtype=torch.float32).div(half))
#     )
#     x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
#     return x.to(dtype=torch.bfloat16)  # Convert back to lower precision




@amp.autocast(enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs

@amp.autocast(enabled=False)
# def rope_apply(x, grid_sizes, freqs):
#     n, c = x.size(2), x.size(3) // 2
#     freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

#     output = []
#     for i, (f, h, w) in enumerate(grid_sizes.tolist()):
#         seq_len = f * h * w

#         x_i = torch.view_as_complex(
#             x[i, :seq_len].to(dtype=torch.float32).reshape(seq_len, n, -1, 2)
#         ).detach()  # Prevent unnecessary gradients
        
#         freqs_i = torch.cat([
#             freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
#             freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
#             freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
#         ], dim=-1).reshape(seq_len, 1, -1).detach()  # Prevent gradient computation

#         x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
#         x_i = torch.cat([x_i, x[i, seq_len:]])

#         output.append(x_i)
    
#     return torch.stack(output).to(dtype=torch.bfloat16)  # Use bfloat16


@amp.autocast(enabled=False)
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x.float()).type_as(x)

# from torch.utils.checkpoint import checkpoint
# class WanSelfAttention(nn.Module):
#     def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
#         assert dim % num_heads == 0
#         super().__init__()
#         self.dim = dim
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.window_size = window_size
#         self.qk_norm = qk_norm
#         self.eps = eps

#         # Use fp16 or bfloat16 parameters
#         self.q = nn.Linear(dim, dim, dtype=torch.bfloat16)
#         self.k = nn.Linear(dim, dim, dtype=torch.bfloat16)
#         self.v = nn.Linear(dim, dim, dtype=torch.bfloat16)
#         self.o = nn.Linear(dim, dim, dtype=torch.bfloat16)

#         self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
#         self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

#     def forward(self, x, seq_lens, grid_sizes, freqs):
#         b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

#         def qkv_fn(x):
#             q = self.norm_q(self.q(x)).view(b, s, n, d).to(dtype=torch.bfloat16)
#             k = self.norm_k(self.k(x)).view(b, s, n, d).to(dtype=torch.bfloat16)
#             v = self.v(x).view(b, s, n, d).to(dtype=torch.bfloat16)
#             return q, k, v

#         q, k, v = qkv_fn(x)

#         x = flash_attention(
#             q=rope_apply(q, grid_sizes, freqs),
#             k=rope_apply(k, grid_sizes, freqs),
#             v=v,
#             k_lens=seq_lens,
#             window_size=self.window_size
#         ).to(dtype=torch.bfloat16)

#         x = x.flatten(2)
#         x = self.o(x)
#         return x


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        x = flash_attention(
            q=rope_apply(q, grid_sizes, freqs),
            k=rope_apply(k, grid_sizes, freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanT2VCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)
        img_x = flash_attention(q, k_img, v_img, k_lens=None)
        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
    'i2v_cross_attn': WanI2VCrossAttention,
}


# class WanAttentionBlock(nn.Module):

#     def __init__(self,
#                  cross_attn_type,
#                  dim,
#                  ffn_dim,
#                  num_heads,
#                  window_size=(-1, -1),
#                  qk_norm=True,
#                  cross_attn_norm=False,
#                  eps=1e-6):
#         super().__init__()
#         self.dim = dim
#         self.ffn_dim = ffn_dim
#         self.num_heads = num_heads
#         self.window_size = window_size
#         self.qk_norm = qk_norm
#         self.cross_attn_norm = cross_attn_norm
#         self.eps = eps

#         # layers
#         self.norm1 = WanLayerNorm(dim, eps)
#         self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)
#         self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
#         self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim, num_heads, (-1, -1), qk_norm, eps)
#         self.norm2 = WanLayerNorm(dim, eps)
#         self.ffn = nn.Sequential(
#             nn.Linear(dim, ffn_dim, bias=False),  # Remove bias for memory efficiency
#             nn.GELU(approximate='tanh'),
#             nn.Linear(ffn_dim, dim, bias=False)   # Remove bias
#         )

#         # modulation
#         self.modulation = nn.Parameter(torch.randn(1, 6, dim, dtype=torch.bfloat16) / dim**0.5)

#     def forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens):
#         assert e.dtype == torch.bfloat16  # Reduce memory with bfloat16
#         e = (self.modulation + e).chunk(6, dim=1)  # Avoid unnecessary autocast

#         # Self-attention
#         y = self.self_attn(self.norm1(x) * (1 + e[1]) + e[0], seq_lens, grid_sizes, freqs)
#         x = x + y * e[2]  

#         # Cross-attention and FFN
#         def cross_attn_ffn(x, context, context_lens, e):
#             x = x + self.cross_attn(self.norm3(x), context, context_lens)
#             y = self.ffn(self.norm2(x) * (1 + e[4]) + e[3])
#             return x + y * e[5]

#         return cross_attn_ffn(x, context, context_lens, e)


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32):
            e = (self.modulation + e).chunk(6, dim=1)
        assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1]) + e[0], seq_lens, grid_sizes,
            freqs)
        with amp.autocast(dtype=torch.float32):
            x = x + y * e[2]

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            y = self.ffn(self.norm2(x).float() * (1 + e[4]) + e[3])
            with amp.autocast(dtype=torch.float32):
                x = x + y * e[5]
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


# class Head(nn.Module):

#     def __init__(self, dim, out_dim, patch_size, eps=1e-6):
#         super().__init__()
#         self.dim = dim
#         self.out_dim = out_dim
#         self.patch_size = patch_size
#         self.eps = eps

#         # layers
#         out_dim = math.prod(patch_size) * out_dim
#         self.norm = WanLayerNorm(dim, eps)
#         self.head = nn.Linear(dim, out_dim, bias=False)  # Remove bias

#         # modulation
#         self.modulation = nn.Parameter(torch.randn(1, 2, dim, dtype=torch.bfloat16) / dim**0.5)

#     def forward(self, x, e):
#         assert e.dtype == torch.bfloat16  
#         e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
#         return self.head(self.norm(x) * (1 + e[1]) + e[0])



class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32):
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        return x


class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 lazy_init=False):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'

        # Initialize model parameters only if lazy_init is False XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        if not lazy_init:
            self.blocks = nn.ModuleList([
                WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                                  window_size, qk_norm, cross_attn_norm, eps)
                for _ in range(num_layers)
            ])
        else:
            # Initialize blocks as None for lazy loading XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
            self.blocks = nn.ModuleList([None for _ in range(num_layers)])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
                               dim=1)

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()


    # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    
    # Including functions for memory optimizations in low VRAM settings

    @classmethod
    def from_pretrained_lazy(cls, checkpoint_dir, device):
        """
        Lazily load the model from disk, block by block, without loading the entire model into memory.

        Args:
            checkpoint_dir (str): Path to the checkpoint directory.
            device (torch.device): Device to load the model onto (e.g., "cuda:0").

        Returns:
            WanModel: A partially loaded model ready for incremental processing.
        """
        # Load the model configuration
        config = cls.load_config(checkpoint_dir)
    
        # Initialize the model structure without allocating parameters
        model = cls(**config, lazy_init=True)  # Pass a flag to avoid parameter allocation
        model.to("cpu")  # Ensure the model is initially on CPU
        model.eval().requires_grad_(False)
    
        # Load the state dict keys to identify blocks
        state_dict_path = os.path.join(checkpoint_dir, "diffusion_pytorch_model.safetensors")
        with safe_open(state_dict_path, framework="pt") as f:
            block_keys = [key for key in f.keys() if "blocks." in key]
    
        # Store the checkpoint directory and device for lazy loading
        model.checkpoint_dir = checkpoint_dir
        model.block_keys = block_keys
        model.state_dict_path = state_dict_path
    
        # Initialize blocks as None (lazy loading)
        for i in range(len(model.blocks)):
            model.blocks[i] = None
    
        return model

    def load_block_from_disk(self, block_idx):
        
        """
        Load a specific block directly from disk to the GPU.
    
        Args:
            block_idx (int): Index of the block to load.
    
        Returns:
            Module: The loaded block.
        """
        # If the block is already loaded, return it
        if self.blocks[block_idx] is not None:
            return self.blocks[block_idx]
    
        # Load the block's state dict from disk
        print("Loading block...")
        block_key_prefix = f"blocks.{block_idx}."
        block_state_dict = {}
    
        with safe_open(self.state_dict_path, framework="pt") as f:
            for key in self.block_keys:
                if key.startswith(block_key_prefix):
                    block_state_dict[key[len(block_key_prefix):]] = f.get_tensor(key)
    
        # Initialize the block and load its state dict

        cross_attn_type = 't2v_cross_attn' if self.model_type == 't2v' else 'i2v_cross_attn'
        block = WanAttentionBlock(
            cross_attn_type,
            dim=self.dim,
            ffn_dim=self.ffn_dim,
            num_heads=self.num_heads,
            window_size=self.window_size,
            qk_norm=self.qk_norm,
            cross_attn_norm=self.cross_attn_norm,
            eps=self.eps
        )
        block.load_state_dict(block_state_dict)

        
    
        # Initialize the block's parameters
        for m in block.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
        # Move the block to the GPU
        block.to(torch.device(f"cuda:{0}"))
    
        # Store the loaded block
        self.blocks[block_idx] = block
        del block_state_dict
        print("Block loaded!")
    
        return block

    def unload_block_from_gpu(self, block_idx):
        """
        Unload a specific block from GPU memory and delete it from CPU memory.
    
        Args:
            block_idx (int): Index of the block to unload.
        """
        block = self.blocks[block_idx]
        if block is not None:
            block.to("cpu")  # Move the block to CPU
            del block  # Delete the block from CPU memory
            
            print("Block used and deleted!")
            self.blocks[block_idx] = None  # Set the block to None to free up the reference

    def process_incremental(self, x, chunks, t, context_cond, seq_len, clip_fea=None, y=None):
        
        # Ensure the model is on the correct device
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)
    
        # Handle conditional inputs (y)
        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
    
        # Embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x])
    
        # Time embeddings
        with amp.autocast(dtype=torch.float32):
            e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32
    
        # Context embeddings
        context_cond = self.text_embedding(
            torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context_cond])
        )
    
        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context_cond = torch.concat([context_clip, context_cond], dim=1)

    
        # Prepare kwargs for conditional and unconditional processing
        kwargs_cond = {
            'e': e0,
            'seq_lens': seq_lens,
            'grid_sizes': grid_sizes,
            'freqs': self.freqs,
            'context': context_cond,
            'context_lens': None,
        }

        # chunkNo = 0
        
        # # Process each chunk of blocks
        # for chunk_indices in chunks:
        #     chunkNo = chunkNo + 1
        #     print (f"processing chunk {chunkNo}...")
        #     for idx in chunk_indices:
        #         # Load the block to GPU
        #         block = self.load_block_from_disk(idx)
        #     for idx in chunk_indices:
        #         print(f"Shape of x before using block {idx}: {x.shape}")
        #         x = block(x, **kwargs_cond)
        #         print(f"Shape of x after block {idx}: {x.shape}")
        #     for idx in chunk_indices:
        #         # Unload the block from GPU
        #         self.unload_block_from_gpu(idx)
        #     torch.cuda.empty_cache()  # Clear GPU cache


        chunkNo = 0
        
        # Process each chunk of blocks
        for chunk_indices in chunks:
            chunkNo += 1
            print(f"Processing chunk {chunkNo}...")
        
            # Estimate memory required for one block
            if chunkNo == 1:  # Estimate memory usage only once
                
                block_memory = self.estimate_block_memory(chunk_indices[0])
                self.unload_block_from_gpu(chunk_indices[0])
                print(f"Estimated memory per block: {block_memory / 1024 ** 2:.2f} MB")
        
            # Calculate available memory
            max_memory = torch.cuda.get_device_properties(0).total_memory  # Total GPU memory
            used_memory = torch.cuda.memory_allocated()  # Currently used memory
            available_memory = max_memory - used_memory
        
            # Calculate the maximum number of blocks that can fit in available memory
            if block_memory > 0:
                max_blocks_per_chunk = max(1, int(available_memory / block_memory))
            else:
                max_blocks_per_chunk = 8
            print(f"Available memory: {available_memory / 1024 ** 3:.2f} GB")
            print(f"Max blocks per chunk: {max_blocks_per_chunk}")
        
            # Split the chunk into smaller sub-chunks if necessary
            sub_chunks = [chunk_indices[i:i + max_blocks_per_chunk] for i in range(0, len(chunk_indices), max_blocks_per_chunk)]
        
            # Process each sub-chunk
            for sub_chunk_indices in sub_chunks:
                # Load all blocks in the sub-chunk
                blocks = []
                for idx in sub_chunk_indices:
                    block = self.load_block_from_disk(idx)
                    blocks.append((idx, block))  # Store block and its index
        
                # Process all blocks in the sub-chunk
                for idx, block in blocks:
                    print(f"Shape of x before using block {idx}: {x.shape}")
                    x = block(x, **kwargs_cond)
                    print(f"Shape of x after block {idx}: {x.shape}")
        
                # Unload all blocks in the sub-chunk
                for idx, block in blocks:
                    self.unload_block_from_gpu(idx)
        
                # Clear GPU cache if memory usage is high
                if torch.cuda.memory_allocated() > 0.8 * max_memory:
                    torch.cuda.empty_cache()
    
        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]  

    def estimate_block_memory(self, indices):
        """
        Estimate the memory usage of a block by measuring the memory allocated before and after loading it.
    
        Args:
            indices (int): Index of the block to estimate memory for.
    
        Returns:
            int: Estimated memory usage in bytes.
        """
        try:
            torch.cuda.empty_cache()  # Clear cache to get accurate measurements
            before_memory = torch.cuda.memory_allocated()
            sample_block = self.load_block_from_disk(indices)
            after_memory = torch.cuda.memory_allocated()
            self.unload_block_from_gpu(indices)
            torch.cuda.empty_cache()  # Clear cache again
            memory_usage = after_memory - before_memory  # Memory used by the block
    
            if memory_usage <= 0:
                raise ValueError("Estimated block memory is zero or negative. This may indicate a measurement error.")
    
            return memory_usage
        except Exception as e:
            print(f"Error estimating block memory: {e}")
            return 8 * 1024 ** 2  # Default to 8 MB if estimation fails

    # ... End of Memory Optimization Functions

    # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        with amp.autocast(dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens)

        for block in self.blocks:
            x = block(x, **kwargs)

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out


    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
