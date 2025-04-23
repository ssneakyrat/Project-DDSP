import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualConvBlock(nn.Module):
    """Lightweight residual convolutional block for temporal feature extraction"""
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

class TemporalAttention(nn.Module):
    """Memory-efficient temporal attention for capturing long-range dependencies"""
    def __init__(self, channels, reduction_factor=8):
        super().__init__()
        self.reduction_factor = reduction_factor
        reduced_dim = max(channels // reduction_factor, 8)
        
        # Efficient projection layers
        self.q_proj = nn.Conv1d(channels, reduced_dim, kernel_size=1)
        self.k_proj = nn.Conv1d(channels, reduced_dim, kernel_size=1)
        self.v_proj = nn.Conv1d(channels, channels, kernel_size=1)
        self.out_proj = nn.Conv1d(channels, channels, kernel_size=1)
        
    def forward(self, x):
        batch_size, channels, seq_len = x.shape
        
        # Project to reduce dimensions for keys and queries
        q = self.q_proj(x)  # B x C/r x T
        k = self.k_proj(x)  # B x C/r x T
        v = self.v_proj(x)  # B x C x T
        
        # Compute attention with reduced dimensions (save memory)
        attn = torch.bmm(
            q.transpose(1, 2),  # B x T x C/r
            k  # B x C/r x T
        ) / (q.size(1) ** 0.5)  # Scale by sqrt(d_k)
        
        attn = F.softmax(attn, dim=-1)  # B x T x T
        
        # Apply attention
        out = torch.bmm(
            attn,  # B x T x T
            v.transpose(1, 2)  # B x T x C
        ).transpose(1, 2)  # B x C x T
        
        # Final projection
        out = self.out_proj(out)
        
        return out

class ExpressionPredictor(nn.Module):
    """
    Lightweight predictor specifically designed for vocal expression parameters.
    
    This module focuses on extracting temporal features related to expressiveness
    like vibrato, breathiness, and formant control using dilated convolutions
    and efficient attention mechanisms.
    """
    def __init__(
            self,
            input_dim=80,           # Default mel spectrogram input dimension
            hidden_dim=48,          # Reduced hidden dimension to save memory
            n_formants=4,           # Number of formants to predict
            n_breath_bands=8,       # Spectral shape bands for breath noise
            use_checkpoint=True     # Enable gradient checkpointing
        ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        
        # Define output dimensions
        self.output_splits = {
            'vibrato_rate': 1,
            'vibrato_depth': 1,
            'formant_freqs': n_formants,
            'formant_bandwidths': n_formants,
            'formant_gains': n_formants,
            'glottal_open_quotient': 1,
            'phase_coherence': 1,
            'breathiness': 1,
            'breath_spectral_shape': n_breath_bands
        }
        
        # Calculate total output dimension
        self.n_out = sum([v for k, v in self.output_splits.items()])
        
        # Ensure hidden_dim is divisible by 4 for GroupNorm compatibility
        hidden_dim = (hidden_dim // 4) * 4
        if hidden_dim < 4:
            hidden_dim = 4
            
        # Store the adjusted hidden_dim
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1),
            nn.GroupNorm(num_groups=min(4, hidden_dim), num_channels=hidden_dim),
            nn.LeakyReLU()
        )
        
        # Multi-scale temporal processing (3 blocks with increasing dilation)
        self.temporal_blocks = nn.ModuleList([
            ResidualConvBlock(hidden_dim, kernel_size=3, dilation=1),
            ResidualConvBlock(hidden_dim, kernel_size=3, dilation=2),
            ResidualConvBlock(hidden_dim, kernel_size=3, dilation=4)
        ])
        
        # Memory-efficient temporal attention
        self.temporal_attention = TemporalAttention(hidden_dim)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(4, hidden_dim), num_channels=hidden_dim),
            nn.LeakyReLU(),
            nn.Conv1d(hidden_dim, self.n_out, kernel_size=1)
        )
    
    def _process_block(self, module, x):
        """Helper for gradient checkpointing"""
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(module, x)
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
    
    def forward(self, mel):
        """
        Args:
            mel: Mel spectrogram [batch_size, n_frames, n_mels]
            
        Returns:
            dict: Dictionary of expression parameters
                  [batch_size, n_frames, parameter_dim]
        """
        # Transpose to channel-first for convolutions
        # [B, T, M] -> [B, M, T]
        x = mel.transpose(1, 2)
        
        # Project input
        x = self.input_proj(x)
        
        # Process through temporal blocks with checkpointing
        for block in self.temporal_blocks:
            x = self._process_block(block, x)
        
        # Apply temporal attention (also with checkpointing if enabled)
        x = self._process_block(self.temporal_attention, x)
        
        # Project to output parameters
        x = self.output_proj(x)
        
        # Transpose back to sequence format [B, C, T] -> [B, T, C]
        x = x.transpose(1, 2)
        
        # Split into parameter dictionary
        expression_params = self.split_to_dict(x)
        
        # Apply activation functions for appropriate ranges
        # Vibrato rate: 0-12 Hz
        expression_params['vibrato_rate'] = 12.0 * torch.sigmoid(expression_params['vibrato_rate'])
        
        # Vibrato depth: 0-1 semitones
        expression_params['vibrato_depth'] = torch.sigmoid(expression_params['vibrato_depth'])
        
        # Formant frequencies: Range depends on position
        # F1: 300-1000 Hz, F2: 800-2500 Hz, F3: 1500-3500 Hz, F4: 2500-4500 Hz
        base_freqs = torch.tensor([300.0, 800.0, 1500.0, 2500.0]).to(mel.device)
        freq_ranges = torch.tensor([700.0, 1700.0, 2000.0, 2000.0]).to(mel.device)
        
        # Process formant frequencies to be in the right ranges
        formant_freqs = expression_params['formant_freqs']
        for i in range(formant_freqs.size(-1)):
            formant_freqs[:, :, i] = base_freqs[i] + freq_ranges[i] * torch.sigmoid(formant_freqs[:, :, i])
        
        # Formant bandwidths: 40-200 Hz
        expression_params['formant_bandwidths'] = 40.0 + 160.0 * torch.sigmoid(expression_params['formant_bandwidths'])
        
        # Formant gains: 0-1 range
        expression_params['formant_gains'] = torch.sigmoid(expression_params['formant_gains'])
        
        # Glottal open quotient: 0.3-0.9 range
        expression_params['glottal_open_quotient'] = 0.3 + 0.6 * torch.sigmoid(expression_params['glottal_open_quotient'])
        
        # Phase coherence: 0-1 range
        expression_params['phase_coherence'] = torch.sigmoid(expression_params['phase_coherence'])
        
        # Breathiness: 0-0.5 range
        expression_params['breathiness'] = 0.5 * torch.sigmoid(expression_params['breathiness'])
        
        # Breath spectral shape: 0-1 range with exponential falloff
        expression_params['breath_spectral_shape'] = torch.sigmoid(expression_params['breath_spectral_shape'])
        
        return expression_params