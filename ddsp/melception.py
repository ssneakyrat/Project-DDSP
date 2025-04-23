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

class TemporalContextModule(nn.Module):
    """Enhanced temporal modeling for capturing dynamics like vibrato and expressions"""
    def __init__(self, channels, context_size=5):
        super().__init__()
        self.context_size = context_size
        
        # Dilated convolutions for multi-scale temporal context
        self.dilated_convs = nn.ModuleList([
            nn.Conv1d(
                channels, channels, 
                kernel_size=3, 
                padding=2**i, 
                dilation=2**i
            ) for i in range(context_size)
        ])
        
        # Attention mechanism to focus on relevant temporal patterns
        self.temporal_attention = nn.Sequential(
            nn.Conv1d(channels * context_size, channels * context_size, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Combine features
        self.combiner = nn.Conv1d(channels * context_size, channels, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        self.act = nn.LeakyReLU()
        
    def forward(self, x):
        # Apply dilated convolutions
        features = []
        for conv in self.dilated_convs:
            features.append(conv(x))
        
        # Concatenate features
        combined = torch.cat(features, dim=1)
        
        # Apply attention
        attention = self.temporal_attention(combined)
        weighted = combined * attention
        
        # Combine and normalize
        out = self.act(self.norm(self.combiner(weighted)))
        
        # Residual connection
        return out + x

class PeriodicPatternExtractor(nn.Module):
    """Specialized module to detect periodic patterns like vibrato"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Multiple kernel sizes to capture different vibrato rates
        self.kernels = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels//6, kernel_size=k, padding=k//2)
            for k in [3, 5, 7, 11, 15, 21]  # Different periods
        ])
        
        self.combiner = nn.Conv1d(out_channels//6 * 6, out_channels, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=min(4, out_channels), num_channels=out_channels)
        self.act = nn.LeakyReLU()
        
    def forward(self, x):
        features = []
        for kernel in self.kernels:
            features.append(kernel(x))
        
        combined = torch.cat(features, dim=1)
        return self.act(self.norm(self.combiner(combined)))

class FormantAwareModule(nn.Module):
    """Specialized module to extract formant information"""
    def __init__(self, channels, n_formants):
        super().__init__()
        self.n_formants = n_formants
        
        # Frequency band analysis with multiple resolutions
        self.band_analysis = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=1),
                nn.GroupNorm(num_groups=min(8, channels), num_channels=channels),
                nn.LeakyReLU(),
                nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=min(channels//8, 8))
            ) for _ in range(4)  # 4 different resolutions
        ])
        
        # Formant peak detection module
        self.peak_detector = nn.Sequential(
            nn.Conv1d(channels*4, channels*2, kernel_size=1),
            nn.GroupNorm(num_groups=min(16, channels*2), num_channels=channels*2),
            nn.LeakyReLU(),
            nn.Conv1d(channels*2, channels, kernel_size=1)
        )
        
    def forward(self, x):
        features = []
        for analyzer in self.band_analysis:
            features.append(analyzer(x))
        
        combined = torch.cat(features, dim=1)
        return self.peak_detector(combined)

class EnhancedMelception(nn.Module):
    """
    Enhanced Melception architecture for mel spectrogram feature extraction with
    expanded capabilities to predict vocal synthesis parameters.
    
    This architecture uses multi-scale parallel convolution paths (inspired by
    Inception networks) along with frequency-aware processing, enhanced temporal
    modeling, and specialized branches for different parameter groups.
    """
    def __init__(
            self,
            input_channel,
            output_splits,
            dim_model=128,  # Increased from 64
            num_blocks=6,   # Increased from 4
            temporal_context_size=5):
        super().__init__()
        self.output_splits = output_splits
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_channel, dim_model, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(8, dim_model), num_channels=dim_model),
            nn.LeakyReLU()
        )
        
        # Main processing blocks with increased capacity
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(MelceptionModule(dim_model))
            
            # Add frequency-aware processing at multiple stages
            if i == num_blocks // 3 - 1 or i == 2*num_blocks // 3 - 1:
                self.blocks.append(FrequencyAwareAttention(dim_model, freq_bands=8))  # More frequency bands
        
        # Enhanced temporal context modeling for vibrato and dynamics
        self.temporal_context = TemporalContextModule(
            dim_model, 
            context_size=temporal_context_size
        )
        
        # Parameter-specific branches
        self.n_out = sum([v for k, v in output_splits.items()])
        
        # Create specialized branches for different parameter groups
        self.create_parameter_branches(dim_model, output_splits)
        
    def create_parameter_branches(self, dim_model, output_splits):
        """Create specialized branches for different parameter types"""
        
        # 1. Basic parameters branch (f0, amplitude)
        basic_outputs = output_splits.get('f0', 0) + output_splits.get('A', 0)
        if basic_outputs > 0:
            self.basic_branch = nn.Sequential(
                nn.Conv1d(dim_model, dim_model//2, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups=min(4, dim_model//2), num_channels=dim_model//2),
                nn.LeakyReLU(),
                nn.Conv1d(dim_model//2, basic_outputs, kernel_size=1)
            )
        
        # 2. Harmonic structure branch (amplitudes, harmonic_magnitude)
        n_harmonic_outputs = output_splits.get('amplitudes', 0) + output_splits.get('harmonic_magnitude', 0)
        if n_harmonic_outputs > 0:
            self.harmonic_branch = nn.Sequential(
                nn.Conv1d(dim_model, dim_model, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups=min(8, dim_model), num_channels=dim_model),
                nn.LeakyReLU(),
                FrequencyAwareAttention(dim_model, freq_bands=16),  # Fine-grained frequency awareness
                nn.Conv1d(dim_model, n_harmonic_outputs, kernel_size=1)
            )
        
        # 3. Noise branch (noise_magnitude, breath_spectral_shape)
        n_noise_outputs = output_splits.get('noise_magnitude', 0) + output_splits.get('breath_spectral_shape', 0)
        if n_noise_outputs > 0:
            self.noise_branch = nn.Sequential(
                nn.Conv1d(dim_model, dim_model//2, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups=min(4, dim_model//2), num_channels=dim_model//2),
                nn.LeakyReLU(),
                nn.Conv1d(dim_model//2, n_noise_outputs, kernel_size=1)
            )
        
        # 4. Vibrato branch with enhanced temporal modeling
        n_vibrato_outputs = output_splits.get('vibrato_rate', 0) + output_splits.get('vibrato_depth', 0)
        if n_vibrato_outputs > 0:
            self.vibrato_branch = nn.Sequential(
                nn.Conv1d(dim_model, dim_model//4, kernel_size=5, padding=2),  # Wider context
                nn.GroupNorm(num_groups=min(2, dim_model//4), num_channels=dim_model//4),
                nn.LeakyReLU(),
                # Specialized temporal convolution for periodic patterns
                PeriodicPatternExtractor(dim_model//4, dim_model//4),
                nn.Conv1d(dim_model//4, n_vibrato_outputs, kernel_size=1)
            )
        
        # 5. Formant branch with frequency-band specific processing
        n_formant_outputs = (output_splits.get('formant_freqs', 0) + 
                            output_splits.get('formant_bandwidths', 0) + 
                            output_splits.get('formant_gains', 0))
        if n_formant_outputs > 0:
            self.formant_branch = nn.Sequential(
                nn.Conv1d(dim_model, dim_model, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups=min(8, dim_model), num_channels=dim_model),
                nn.LeakyReLU(),
                FormantAwareModule(dim_model, output_splits.get('formant_freqs', 4)),
                nn.Conv1d(dim_model, n_formant_outputs, kernel_size=1)
            )
        
        # 6. Voice quality branch (glottal_open_quotient, phase_coherence, breathiness)
        n_quality_outputs = (output_splits.get('glottal_open_quotient', 0) + 
                            output_splits.get('phase_coherence', 0) + 
                            output_splits.get('breathiness', 0))
        if n_quality_outputs > 0:
            self.quality_branch = nn.Sequential(
                nn.Conv1d(dim_model, dim_model//2, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups=min(4, dim_model//2), num_channels=dim_model//2),
                nn.LeakyReLU(),
                nn.Conv1d(dim_model//2, n_quality_outputs, kernel_size=1)
            )
    
    def forward(self, x):
        """
        input: 
            B x n_frames x n_mels
        return: 
            dict of B x n_frames x feat
        """
        # Transpose to channel-first format for convolutions
        # x: B x n_frames x n_mels -> B x n_mels x n_frames
        x = x.transpose(1, 2)
        
        # Input projection
        x = self.input_proj(x)
        
        # Process through all blocks
        for block in self.blocks:
            x = block(x)
        
        # Enhanced temporal context
        x = self.temporal_context(x)
        
        # Process through specialized branches
        outputs = {}
        
        # Collect outputs from each branch
        if hasattr(self, 'basic_branch'):
            basic_out = self.basic_branch(x)
            f0_size = self.output_splits.get('f0', 0)
            if f0_size > 0:
                outputs['f0'] = basic_out[:, :f0_size, :]
                outputs['A'] = basic_out[:, f0_size:, :]
            else:
                outputs['A'] = basic_out
        
        # Harmonic structure
        if hasattr(self, 'harmonic_branch'):
            harmonic_out = self.harmonic_branch(x)
            split_idx = self.output_splits.get('amplitudes', 0)
            if split_idx > 0:
                outputs['amplitudes'] = harmonic_out[:, :split_idx, :]
                outputs['harmonic_magnitude'] = harmonic_out[:, split_idx:, :]
            else:
                outputs['harmonic_magnitude'] = harmonic_out
        
        # Noise characteristics
        if hasattr(self, 'noise_branch'):
            noise_out = self.noise_branch(x)
            split_idx = self.output_splits.get('noise_magnitude', 0)
            if split_idx > 0:
                outputs['noise_magnitude'] = noise_out[:, :split_idx, :]
                outputs['breath_spectral_shape'] = noise_out[:, split_idx:, :]
            else:
                outputs['noise_magnitude'] = noise_out
        
        # Vibrato parameters
        if hasattr(self, 'vibrato_branch') and self.output_splits.get('vibrato_rate', 0) > 0:
            vibrato_out = self.vibrato_branch(x)
            split_idx = self.output_splits.get('vibrato_rate', 0)
            outputs['vibrato_rate'] = vibrato_out[:, :split_idx, :]
            outputs['vibrato_depth'] = vibrato_out[:, split_idx:, :]
        
        # Formant parameters
        if hasattr(self, 'formant_branch'):
            formant_out = self.formant_branch(x)
            idx1 = self.output_splits.get('formant_freqs', 0)
            idx2 = idx1 + self.output_splits.get('formant_bandwidths', 0)
            if idx1 > 0:
                outputs['formant_freqs'] = formant_out[:, :idx1, :]
                outputs['formant_bandwidths'] = formant_out[:, idx1:idx2, :]
                outputs['formant_gains'] = formant_out[:, idx2:, :]
        
        # Voice quality parameters
        if hasattr(self, 'quality_branch'):
            quality_out = self.quality_branch(x)
            idx1 = self.output_splits.get('glottal_open_quotient', 0)
            idx2 = idx1 + self.output_splits.get('phase_coherence', 0)
            if idx1 > 0:
                outputs['glottal_open_quotient'] = quality_out[:, :idx1, :]
                outputs['phase_coherence'] = quality_out[:, idx1:idx2, :]
                outputs['breathiness'] = quality_out[:, idx2:, :]
        
        # Transpose all outputs back to sequence format (B x n_frames x feat)
        for key in outputs:
            outputs[key] = outputs[key].transpose(1, 2)
        
        return outputs


# Legacy class for backward compatibility
class Melception(EnhancedMelception):
    """
    Original Melception architecture for mel spectrogram feature extraction.
    
    This class inherits from EnhancedMelception for backward compatibility.
    It implements the original forward method that returns a dictionary split
    based on the output_splits parameter.
    """
    def __init__(
            self,
            input_channel,
            output_splits,
            dim_model=64,
            num_blocks=4):
        super().__init__(
            input_channel=input_channel,
            output_splits=output_splits,
            dim_model=dim_model,
            num_blocks=num_blocks,
            temporal_context_size=3
        )
        
    def forward(self, x):
        '''
        input: 
            B x n_frames x n_mels
        return: 
            dict of B x n_frames x feat
        '''
        outputs = super().forward(x)
        
        # Original behavior - split the output tensor into a dictionary
        # This maintains backward compatibility with the original interface
        if '_legacy_output' in self.output_splits:
            # Legacy mode: Return a single combined tensor
            combined_out = torch.cat([outputs[key] for key in self.output_splits if key != '_legacy_output'], dim=-1)
            return split_to_dict(combined_out, self.output_splits)
            
        return outputs