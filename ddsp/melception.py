import torch
import torch.nn as nn
import torch.nn.functional as F

def split_to_dict(tensor, tensor_splits):
    """Split a tensor into a dictionary of multiple tensors."""
    labels = []
    sizes = []

    for k, v in tensor_splits.items():
        labels.append(k)
        sizes.append(v)

    tensors = torch.split(tensor, sizes, dim=-1)
    return dict(zip(labels, tensors))

class SqueezeExcitation(nn.Module):
    """Channel attention module that highlights important features."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class MultiScalePath(nn.Module):
    """Single path in the multi-scale parallel architecture with specific kernel size."""
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.norm1 = nn.GroupNorm(num_groups=min(4, out_channels), num_channels=out_channels)
        self.norm2 = nn.GroupNorm(num_groups=min(4, out_channels), num_channels=out_channels)
        self.act = nn.LeakyReLU()
        
    def forward(self, x):
        x = self.act(self.norm1(self.conv1(x)))
        x = self.act(self.norm2(self.conv2(x)))
        return x

class MelceptionModule(nn.Module):
    """Core building block with multi-scale parallel paths inspired by Inception."""
    def __init__(self, channels):
        super().__init__()
        # 1x1 convolution path
        self.path1 = nn.Sequential(
            nn.Conv1d(channels, channels//4, kernel_size=1),
            nn.GroupNorm(num_groups=min(4, channels//4), num_channels=channels//4),
            nn.LeakyReLU()
        )
        
        # 3x3 convolution path
        self.path2 = MultiScalePath(channels, channels//4, kernel_size=3)
        
        # 5x5 convolution path
        self.path3 = MultiScalePath(channels, channels//4, kernel_size=5)
        
        # 7x7 convolution path
        self.path4 = MultiScalePath(channels, channels//4, kernel_size=7)
        
        # Channel attention
        self.se = SqueezeExcitation(channels)
        
        # Dimension reduction after concatenation
        self.combiner = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=min(4, channels), num_channels=channels)
        self.act = nn.LeakyReLU()
        
        # Residual connection
        self.residual = nn.Conv1d(channels, channels, kernel_size=1)
        
    def forward(self, x):
        residual = self.residual(x)
        
        # Process through all paths
        path1 = self.path1(x)
        path2 = self.path2(x)
        path3 = self.path3(x)
        path4 = self.path4(x)
        
        # Concatenate all paths
        combined = torch.cat([path1, path2, path3, path4], dim=1)
        
        # Apply channel attention
        combined = self.se(combined)
        
        # Dimension reduction
        out = self.act(self.norm(self.combiner(combined)))
        
        # Residual connection
        return out + residual

class DownsampledAttention(nn.Module):
    """Efficient attention mechanism that works on downsampled representation."""
    def __init__(self, channels, reduction_factor=8):
        super().__init__()
        self.reduction_factor = reduction_factor
        
        # Downsample to fixed length for efficiency
        self.pool = nn.AdaptiveAvgPool1d(reduction_factor)
        
        # Simple self-attention on the pooled representation
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)
        
    def forward(self, x):
        b, c, t = x.shape
        
        # Pool to fixed length
        pooled = self.pool(x)  # B x C x R
        pooled = pooled.transpose(1, 2)  # B x R x C
        
        # Self-attention
        q = self.q_proj(pooled)
        k = self.k_proj(pooled)
        v = self.v_proj(pooled)
        
        # Scaled dot-product attention
        attention = torch.bmm(q, k.transpose(1, 2)) / (c ** 0.5)
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention weights
        context = torch.bmm(attention, v)
        out = self.out_proj(context)  # B x R x C
        
        # Interpolate back to original length
        out = out.transpose(1, 2)  # B x C x R
        out = F.interpolate(out, size=t, mode='linear')  # B x C x T
        
        return out

class FrequencyAwareAttention(nn.Module):
    """Attention module that operates on frequency bands."""
    def __init__(self, channels, freq_bands=4):
        super().__init__()
        self.freq_bands = freq_bands
        channels_per_band = channels // freq_bands
        
        # Process each frequency band separately
        self.band_processors = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(channels_per_band, channels_per_band, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups=min(4, channels_per_band), num_channels=channels_per_band),
                nn.LeakyReLU(),
                SqueezeExcitation(channels_per_band)
            ) for _ in range(freq_bands)
        ])
        
        # Frequency band attention
        self.band_attention = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, t = x.shape
        band_size = c // self.freq_bands
        
        # Split into frequency bands
        bands = torch.split(x, band_size, dim=1)
        
        # Process each band
        processed_bands = [processor(band) for band, processor in zip(bands, self.band_processors)]
        
        # Recombine bands
        recombined = torch.cat(processed_bands, dim=1)
        
        # Apply band attention
        attention = self.band_attention(recombined)
        
        return recombined * attention

class Melception(nn.Module):
    """
    Melception architecture for mel spectrogram feature extraction.
    
    This architecture uses multi-scale parallel convolution paths (inspired by
    Inception networks) along with frequency-aware processing and efficient
    attention mechanisms for mel spectrograms.
    """
    def __init__(
            self,
            input_channel,
            output_splits,
            dim_model=64,
            num_blocks=4):
        super().__init__()
        self.output_splits = output_splits
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_channel, dim_model, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(4, dim_model), num_channels=dim_model),
            nn.LeakyReLU()
        )
        
        # Main processing blocks
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(MelceptionModule(dim_model))
            
            # Add frequency-aware processing after half the blocks
            if i == num_blocks // 2 - 1:
                self.blocks.append(FrequencyAwareAttention(dim_model))
        
        # Global context attention (on downsampled representation for efficiency)
        self.global_context = DownsampledAttention(dim_model)
        
        # Output projection
        self.n_out = sum([v for k, v in output_splits.items()])
        self.output_proj = nn.Sequential(
            nn.Conv1d(dim_model, dim_model, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(4, dim_model), num_channels=dim_model),
            nn.LeakyReLU(),
            nn.Conv1d(dim_model, self.n_out, kernel_size=1)
        )
        
    def forward(self, x):
        '''
        input: 
            B x n_frames x n_mels
        return: 
            dict of B x n_frames x feat
        '''
        # Transpose to channel-first format for convolutions
        # x: B x n_frames x n_mels -> B x n_mels x n_frames
        x = x.transpose(1, 2)
        
        # Input projection
        x = self.input_proj(x)
        
        # Process through all blocks
        for block in self.blocks:
            x = block(x)
        
        # Apply global context attention
        context = self.global_context(x)
        x = x + context
        
        # Output projection
        x = self.output_proj(x)
        
        # Transpose back to sequence format
        # x: B x n_out x n_frames -> B x n_frames x n_out
        x = x.transpose(1, 2)
        
        # Split to output dict (same interface as PCmer)
        controls = split_to_dict(x, self.output_splits)
        
        return controls
