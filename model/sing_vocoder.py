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
        qkv = self.qkv(x)  # [b, 3*c, t]
        
        # Reshape: [b, 3*c, t] -> [b, 3, heads, head_dim, t]
        qkv = qkv.reshape(b, 3, self.heads, self.head_dim, t)
        
        # Split into q, k, v
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # Each has shape [b, heads, head_dim, t]
        
        # Transpose q and v for attention calculation
        q = q.transpose(-2, -1)  # [b, heads, t, head_dim]
        # No transpose for k as it's already in the correct shape [b, heads, head_dim, t]
        v = v.transpose(-2, -1)  # [b, heads, t, head_dim]
        
        # Compute attention scores
        attn = torch.matmul(q, k) * self.scale  # [b, heads, t, t]
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention
        out = torch.matmul(attn, v)  # [b, heads, t, head_dim]
        
        # Reshape back to original format: [b, heads, t, head_dim] -> [b, c, t]
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
    
    Enhanced with explicit formant modeling and improved singer adaptation.
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
            n_formants=4,  # Number of formants to model
            use_checkpoint=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.n_formants = n_formants
        
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
        
        # Core parameter outputs with new formant and singer adaptation parameters
        self.output_splits = {
            'f0': 1,
            'A': 1,
            'amplitudes': n_harmonics,
            'harmonic_magnitude': n_mag_harmonic,
            'noise_magnitude': n_mag_noise,
            # New formant-specific parameters
            'formant_freqs': n_formants,      # Center frequencies for each formant
            'formant_widths': n_formants,     # Bandwidth for each formant 
            'formant_gains': n_formants,      # Gain for each formant
            # Vibrato parameters for enhanced expression
            'vibrato_rate': 1,                # Rate of vibrato in Hz
            'vibrato_depth': 1,               # Depth of vibrato in cents
            'vibrato_delay': 1,               # Onset delay of vibrato
            # Voice quality parameters
            'breathiness': 1,                 # Breathiness control
            'tension': 1,                     # Vocal tension control
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
        
        # Singer adaptation networks
        self.formant_shift = nn.Linear(self.singer_embed_dim, n_formants)
        self.timbre_control = nn.Linear(self.singer_embed_dim, n_mag_harmonic)
        self.breathiness_control = nn.Linear(self.singer_embed_dim, 1)
        self.tension_control = nn.Linear(self.singer_embed_dim, 1)
        self.vibrato_style = nn.Linear(self.singer_embed_dim, 3)  # rate, depth, delay
    
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
    
    def get_singer_embedding(self, singer_id=None, singer_weights=None, singer_ids=None):
        """
        Get singer embedding with support for single singer or mixed singers
        
        Args:
            singer_id: Single singer ID [batch_size, 1]
            singer_weights: Weights for mixing multiple singers [batch_size, n_singers]
            singer_ids: IDs for mixing multiple singers [batch_size, n_singers]
            
        Returns:
            singer_embed: Singer embedding [batch_size, singer_embed_dim]
        """
        if singer_weights is not None and singer_ids is not None:
            # Mix multiple singers
            batch_size, n_singers = singer_weights.shape
            
            # Get embeddings for all singers
            all_embeddings = []
            for i in range(n_singers):
                # Handle different possible shapes
                if singer_ids.dim() == 3:  # [B, n_singers, 1]
                    singer_id_i = singer_ids[:, i, :]
                else:  # [B, n_singers]
                    singer_id_i = singer_ids[:, i:i+1]
                
                singer_embed_i = self.singer_embedding(singer_id_i)  # [B, 1, embed_dim]
                
                # Ensure correct dimensionality
                if singer_embed_i.dim() == 3:
                    singer_embed_i = singer_embed_i.squeeze(1)  # [B, embed_dim]
                elif singer_embed_i.dim() == 4:
                    singer_embed_i = singer_embed_i.squeeze(1).squeeze(1)  # Handle [B, 1, 1, embed_dim]
                
                all_embeddings.append(singer_embed_i)
            
            # Stack and apply weights
            stacked_embeddings = torch.stack(all_embeddings, dim=1)  # [batch, n_singers, embed_dim]
            singer_weights = singer_weights.unsqueeze(-1)  # [batch, n_singers, 1]
            weighted_embeddings = stacked_embeddings * singer_weights
            
            # Combine to get final embedding
            singer_embed = weighted_embeddings.sum(dim=1)  # [batch, embed_dim]
        else:
            # Single singer mode - handle different possible shapes
            singer_embed = self.singer_embedding(singer_id)  # Could be [B, 1, embed_dim] or [B, 1, 1, embed_dim]
            
            # Ensure output is [B, embed_dim] by removing extra dimensions
            if singer_embed.dim() == 3:
                singer_embed = singer_embed.squeeze(1)  # [B, embed_dim]
            elif singer_embed.dim() == 4:
                singer_embed = singer_embed.squeeze(1).squeeze(1)  # Handle [B, 1, 1, embed_dim]
        
        return singer_embed
    
    def apply_singer_adaptation(self, params, singer_embed):
        """
        Apply singer-specific adaptations to parameters
        
        Args:
            params: Dictionary of parameters
            singer_embed: Singer embedding [batch_size, singer_embed_dim]
            
        Returns:
            params: Adapted parameters
        """
        # Ensure singer_embed has correct shape [batch_size, singer_embed_dim]
        if singer_embed.dim() > 2:
            singer_embed = singer_embed.reshape(singer_embed.size(0), -1)
        
        # Formant shift adaptation (scaled with tanh for ±50% range)
        formant_shift_factors = torch.tanh(self.formant_shift(singer_embed)) * 0.5  # [B, n_formants]
        formant_shift_factors = formant_shift_factors.unsqueeze(1)  # [B, 1, n_formants]
        params['formant_freqs'] = params['formant_freqs'] * (1.0 + formant_shift_factors)
        
        # Timbre control adaptation
        timbre_weights = torch.sigmoid(self.timbre_control(singer_embed))  # [B, n_mag_harmonic]
        timbre_weights = timbre_weights.unsqueeze(1)  # [B, 1, n_mag_harmonic]
        params['harmonic_magnitude'] = params['harmonic_magnitude'] * timbre_weights
        
        # Breathiness adaptation
        breathiness = torch.sigmoid(self.breathiness_control(singer_embed))  # [B, 1]
        breathiness = breathiness.unsqueeze(1)  # [B, 1, 1]
        params['breathiness'] = params['breathiness'] * breathiness
        params['noise_magnitude'] = params['noise_magnitude'] * (0.5 + 0.5 * params['breathiness'])
        
        # Tension adaptation
        tension = torch.sigmoid(self.tension_control(singer_embed))  # [B, 1]
        tension = tension.unsqueeze(1)  # [B, 1, 1]
        params['tension'] = params['tension'] * tension
        
        # Vibrato style adaptation
        vibrato_style = self.vibrato_style(singer_embed)  # [B, 3]
        vibrato_style = vibrato_style.unsqueeze(1)  # [B, 1, 3]
        params['vibrato_rate'] = params['vibrato_rate'] * (0.5 + 0.5 * torch.sigmoid(vibrato_style[:, :, 0:1]))
        params['vibrato_depth'] = params['vibrato_depth'] * (0.5 + 0.5 * torch.sigmoid(vibrato_style[:, :, 1:2]))
        params['vibrato_delay'] = params['vibrato_delay'] * (0.5 + 0.5 * torch.sigmoid(vibrato_style[:, :, 2:3]))
        
        return params
    
    def forward(self, phone_seq, f0, singer_id=None, language_id=None, 
                singer_weights=None, singer_ids=None):
        """
        Forward pass with enhanced formant modeling and singer adaptation
        
        Args:
            phone_seq: Phoneme sequence [batch_size, n_frames] (Long tensor of phoneme IDs)
            f0: Fundamental frequency [batch_size, n_frames, 1]
            singer_id: Singer IDs [batch_size, 1] (Long tensor of singer IDs)
            language_id: Language IDs [batch_size, 1] (Long tensor of language IDs)
            singer_weights: Optional weights for mixing singers [batch_size, n_singers]
            singer_ids: Optional IDs for mixing singers [batch_size, n_singers]
        
        Returns:
            dict: Dictionary of synthesis parameters [batch_size, n_frames, param_dim]
        """
        batch_size, n_frames = phone_seq.shape
        
        # Embed phonemes
        phone_emb = self.phone_embedding(phone_seq)  # [B, T, phone_dim]
        
        # Get singer embedding (with support for mixing)
        singer_emb = self.get_singer_embedding(singer_id, singer_weights, singer_ids)  # [B, singer_dim]
        
        # Properly reshape for sequence expansion
        singer_emb = singer_emb.unsqueeze(1)  # [B, 1, singer_dim]
        singer_emb = singer_emb.expand(-1, n_frames, -1)  # [B, T, singer_dim]
        
        # Embed language (expand to match sequence length)
        language_emb = self.language_embedding(language_id)  # [B, 1, language_dim]
        
        # Handle potential extra dimensions
        if language_emb.dim() > 3:
            language_emb = language_emb.squeeze(1)  # Remove extra dimension if present
            
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
        
        # Apply default frequency range scaling to formant frequencies
        # Default formant ranges are based on typical human voice formants
        # F1: 300-800 Hz, F2: 800-2500 Hz, F3: 2500-3500 Hz, F4: 3500-4500 Hz
        formant_base_freqs = torch.tensor([500.0, 1500.0, 2500.0, 3500.0][:self.n_formants])
        formant_base_freqs = formant_base_freqs.to(x.device)
        formant_base_freqs = formant_base_freqs.view(1, 1, -1)
        
        # Scale sigmoid output to appropriate formant ranges
        params['formant_freqs'] = torch.sigmoid(params['formant_freqs']) * 2.0 + 0.5  # Range: 0.5-2.5
        params['formant_freqs'] = params['formant_freqs'] * formant_base_freqs
        
        # Scale formant widths (bandwidths) to appropriate ranges (typically 50-300 Hz)
        params['formant_widths'] = torch.sigmoid(params['formant_widths']) * 250.0 + 50.0
        
        # Scale formant gains to reasonable dB range (-10 to +10 dB)
        params['formant_gains'] = torch.tanh(params['formant_gains']) * 10.0
        
        # Apply singer-specific adaptations
        singer_embed_for_adaptation = self.get_singer_embedding(singer_id, singer_weights, singer_ids)
        params = self.apply_singer_adaptation(params, singer_embed_for_adaptation)
        
        return params