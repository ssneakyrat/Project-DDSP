import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

class ResidualConvBlock(nn.Module):
    """Residual convolutional block for temporal feature extraction"""
    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size=kernel_size, 
            padding=(kernel_size-1)//2 * dilation, dilation=dilation
        )
        
        # Ensure num_groups divides num_channels for GroupNorm
        num_groups = min(4, channels)
        if channels % num_groups != 0:
            for i in range(num_groups, 0, -1):
                if channels % i == 0:
                    num_groups = i
                    break
                    
        self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size=kernel_size, 
            padding=(kernel_size-1)//2 * dilation, dilation=dilation
        )
        self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        self.act = nn.LeakyReLU()
        
    def forward(self, x):
        residual = x
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + residual)

class SelfAttention(nn.Module):
    """Self-attention module for capturing long-range dependencies"""
    def __init__(self, channels, heads=4):
        super().__init__()
        self.heads = heads
        assert channels % heads == 0, "Channels must be divisible by number of heads"
        
        self.head_dim = channels // heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(min(4, channels), channels)
        
    def forward(self, x):
        b, c, t = x.shape
        residual = x
        
        # Compute query, key, value
        qkv = self.qkv(x)
        qkv = qkv.reshape(b, 3, self.heads, self.head_dim, t)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        
        # Compute attention scores
        q = q.transpose(-2, -1)  # b, heads, t, head_dim
        k = k.permute(0, 1, 3, 2)  # b, heads, head_dim, t
        v = v.transpose(-2, -1)  # b, heads, t, head_dim
        
        attn = torch.matmul(q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention
        out = torch.matmul(attn, v)  # b, heads, t, head_dim
        out = out.transpose(2, 3).reshape(b, c, t)
        
        # Project and add residual
        out = self.proj(out)
        return self.norm(out) + residual

class SingVocoder(nn.Module):
    """
    Neural network for predicting synthesis parameters from linguistic inputs.
    Replaces Melception and ExpressionPredictor with a single unified network.
    
    Takes phoneme sequence, phoneme durations, f0, singer_id, and language_id as input
    and outputs core synthesis parameters.
    """
    def __init__(
            self,
            phone_map_len, 
            singer_map_len, 
            language_map_len,
            hidden_dim=256,
            n_harmonics=150,
            n_mag_harmonic=256,
            n_mag_noise=80,
            use_checkpoint=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        
        # Embedding dimensions
        self.phone_embed_dim = 128
        self.singer_embed_dim = 64
        self.language_embed_dim = 32
        
        # Embedding layers
        self.phone_embedding = nn.Embedding(phone_map_len, self.phone_embed_dim)
        self.singer_embedding = nn.Embedding(singer_map_len, self.singer_embed_dim)
        self.language_embedding = nn.Embedding(language_map_len, self.language_embed_dim)
        
        # Calculate combined dimension
        combined_dim = self.phone_embed_dim + self.singer_embed_dim + self.language_embed_dim + 1  # +1 for f0
        
        # Input projection to hidden dimension
        self.input_proj = nn.Sequential(
            nn.Conv1d(combined_dim, hidden_dim, kernel_size=1),
            nn.GroupNorm(min(4, hidden_dim), hidden_dim),
            nn.LeakyReLU()
        )
        
        # Temporal encoder with dilated convolutions
        self.conv_blocks = nn.ModuleList([
            ResidualConvBlock(hidden_dim, kernel_size=3, dilation=1),
            ResidualConvBlock(hidden_dim, kernel_size=3, dilation=2),
            ResidualConvBlock(hidden_dim, kernel_size=3, dilation=4),
            ResidualConvBlock(hidden_dim, kernel_size=3, dilation=8),
            ResidualConvBlock(hidden_dim, kernel_size=3, dilation=16),
            ResidualConvBlock(hidden_dim, kernel_size=3, dilation=32)
        ])
        
        # Self-attention module
        self.self_attention = SelfAttention(hidden_dim)
        
        # Bi-directional GRU for temporal coherence
        self.gru = nn.GRU(
            hidden_dim, 
            hidden_dim // 2,
            bidirectional=True,
            batch_first=True
        )
        
        # Core parameter outputs
        self.output_splits = {
            'f0': 1,
            'A': 1,
            'amplitudes': n_harmonics,
            'harmonic_magnitude': n_mag_harmonic,
            'noise_magnitude': n_mag_noise
        }
        
        # Calculate total output dimension
        self.n_out = sum([v for k, v in self.output_splits.items()])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(min(4, hidden_dim), hidden_dim),
            nn.LeakyReLU(),
            nn.Conv1d(hidden_dim, self.n_out, kernel_size=1)
        )
    
    def _apply_checkpoint(self, module, x):
        """Apply gradient checkpointing if enabled"""
        if self.use_checkpoint and self.training:
            return checkpoint.checkpoint(module, x)
        else:
            return module(x)
    
    def split_to_dict(self, tensor):
        """Split tensor into dictionary based on output_splits"""
        outputs = {}
        start = 0
        
        for key, dim in self.output_splits.items():
            outputs[key] = tensor[:, :, start:start+dim]
            start += dim
            
        return outputs
    
    def forward(self, phone_seq, f0, singer_id, language_id):
        """
        Args:
            phone_seq: Phoneme sequence [batch_size, n_frames] (Long tensor of phoneme IDs)
            f0: Fundamental frequency [batch_size, n_frames, 1]
            singer_id: Singer IDs [batch_size, 1] (Long tensor of singer IDs)
            language_id: Language IDs [batch_size, 1] (Long tensor of language IDs)
        
        Returns:
            dict: Dictionary of synthesis parameters [batch_size, n_frames, param_dim]
        """
        batch_size, n_frames = phone_seq.shape
        
        # Embed phonemes
        phone_emb = self.phone_embedding(phone_seq)  # [B, T, phone_dim]
        
        # Embed singer (expand to match sequence length)
        singer_emb = self.singer_embedding(singer_id)  # [B, 1, singer_dim]
        singer_emb = singer_emb.expand(-1, n_frames, -1)  # [B, T, singer_dim]
        
        # Embed language (expand to match sequence length)
        language_emb = self.language_embedding(language_id)  # [B, 1, language_dim]
        language_emb = language_emb.expand(-1, n_frames, -1)  # [B, T, language_dim]
        
        if f0.dim() == 2:  # If f0 is [B, T]
            f0 = f0.unsqueeze(-1)  # Make it [B, T, 1]

        # Concatenate inputs
        combined = torch.cat([
            phone_emb,         # [B, T, phone_dim]
            singer_emb,        # [B, T, singer_dim]
            language_emb,      # [B, T, language_dim]
            f0                 # [B, T, 1]
        ], dim=-1)  # [B, T, combined_dim]
        
        # Transpose for convolution (B, T, C) -> (B, C, T)
        x = combined.transpose(1, 2)
        
        # Apply input projection
        x = self.input_proj(x)
        
        # Apply conv blocks with checkpointing
        for block in self.conv_blocks:
            x = self._apply_checkpoint(block, x)
        
        # Apply self-attention
        x = self._apply_checkpoint(self.self_attention, x)
        
        # Apply GRU (need to transpose back to batch-first)
        x_trans = x.transpose(1, 2)  # [B, T, C]
        
        if self.use_checkpoint and self.training:
            # Custom checkpointing for GRU
            def run_gru(x_gru):
                return self.gru(x_gru)[0]
            
            x_gru = checkpoint.checkpoint(run_gru, x_trans)
        else:
            x_gru, _ = self.gru(x_trans)
        
        # Transpose back to channel-first
        x = x_gru.transpose(1, 2)  # [B, C, T]
        
        # Apply output projection
        x = self._apply_checkpoint(self.output_proj, x)
        
        # Transpose back to sequence format [B, C, T] -> [B, T, C]
        x = x.transpose(1, 2)
        
        # Split into parameter dictionary
        params = self.split_to_dict(x)
        
        return params