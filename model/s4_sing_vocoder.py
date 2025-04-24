import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class S4Core(nn.Module):
    """
    Core S4 (Structured State Space Sequence) implementation.
    This is a simplified version of the S4 layer that focuses on efficiency.
    """
    def __init__(self, d_state=64, d_model=128, bidirectional=True):
        super().__init__()
        self.d_state = d_state
        self.d_model = d_model
        self.bidirectional = bidirectional
        
        # SSM parameters
        # Using complex normal initialization for stability
        self.Lambda = nn.Parameter(torch.randn(self.d_state) + 1j * torch.randn(self.d_state))
        self.Lambda_re = nn.Parameter(-0.5 * torch.ones(self.d_state))
        self.Lambda_im = nn.Parameter(torch.arange(self.d_state).float() * 2 * math.pi / self.d_state)
        
        # Input projection B
        self.B = nn.Parameter(torch.randn(self.d_state, 1))
        
        # Output projection C
        self.C = nn.Parameter(torch.randn(1, self.d_state, 2))  # complex parameters
        
        # Direct term (skip connection within SSM)
        self.D = nn.Parameter(torch.randn(1))
    
    def get_Lambda(self):
        """Returns the diagonal state matrix Lambda as a complex parameter"""
        return torch.complex(
            self.Lambda_re,
            self.Lambda_im
        )
    
    def forward(self, u, return_state=False):
        """
        Forward pass with cached convolution implementation.
        
        Args:
            u: input sequence [B, L, H]
            return_state: whether to return the internal state
            
        Returns:
            y: output sequence [B, L, H]
        """
        # Get parameters and dimensions
        Lambda = self.get_Lambda()  # [d_state]
        B = self.B  # [d_state, 1]
        C = torch.view_as_complex(self.C.permute(2, 0, 1).contiguous())  # [1, d_state]
        D = self.D
        L = u.size(1)
        
        # Compute SSM kernel using cached FFT-based convolution
        # k(t) = C*exp(Lambda*t)*B
        omega = torch.arange(L, device=u.device).float()
        # Discretize the continuous-time system
        Lambda_k = torch.exp(Lambda.unsqueeze(-1) * omega.unsqueeze(0))  # [d_state, L]
        k = torch.matmul(C, Lambda_k * B.unsqueeze(-1)).squeeze()  # [L]
        
        # Convert to real
        k = torch.real(k)
        
        # If bidirectional, create a flipped kernel as well
        if self.bidirectional:
            k_flip = torch.flip(k, [0])
            k = torch.cat([k, k_flip])
        
        # Reshape kernel for efficient batch convolution
        k_f = torch.fft.rfft(k.float(), n=2*L)
        
        # Apply convolution in Fourier domain for efficiency
        u_f = torch.fft.rfft(u.float(), n=2*L)
        y_f = u_f * k_f.unsqueeze(0).unsqueeze(-1)
        y = torch.fft.irfft(y_f, n=2*L)[..., :L]
        
        # Add direct bypass term
        y = y + u * D
        
        if return_state:
            return y, None  # Simplified - not returning actual state
        else:
            return y


class S4Layer(nn.Module):
    """
    S4 layer with normalization and gating
    """
    def __init__(self, d_model=128, d_state=64, dropout=0.1, activation='gelu', bidirectional=True):
        super().__init__()
        
        # Normalization and dropout
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # S4 block
        self.d_model = d_model
        self.S4 = nn.ModuleList([
            S4Core(d_state=d_state, d_model=d_model, bidirectional=bidirectional)
            for _ in range(d_model)
        ])
        
        # Activation
        if activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'relu':
            self.activation = F.relu
        else:
            self.activation = F.silu  # SiLU/Swish
    
    def forward(self, x):
        """
        Args:
            x: [B, L, D]
        """
        # Apply normalization
        z = self.norm(x)
        
        # Apply S4 blocks (one per feature dimension)
        # This is the independent SSM setup for maximum parallelization
        out = []
        for i in range(self.d_model):
            out.append(self.S4[i](z[..., i:i+1]))
        y = torch.cat(out, dim=-1)
        
        # Apply activation and dropout
        y = self.activation(y)
        y = self.dropout(y)
        
        # Residual connection
        return x + y


class ParallelS4Layer(nn.Module):
    """
    More efficient parallel implementation of S4 layer
    Uses a single S4 core with wider state instead of multiple S4 cores
    """
    def __init__(self, d_model=128, d_state=64, dropout=0.1, activation='gelu', bidirectional=True, 
                 chunk_size=4):
        super().__init__()
        
        # Normalization and dropout
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # S4 parameters
        self.d_model = d_model
        self.chunk_size = chunk_size
        
        # Create chunked S4 cores for better parallelism
        self.num_chunks = math.ceil(d_model / chunk_size)
        self.S4_parallel = nn.ModuleList([
            S4Core(
                d_state=d_state, 
                d_model=min(chunk_size, d_model - i * chunk_size),
                bidirectional=bidirectional
            )
            for i in range(self.num_chunks)
        ])
        
        # In projections and out projections
        self.in_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)
        
        # Activation
        if activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'relu':
            self.activation = F.relu
        else:
            self.activation = F.silu  # SiLU/Swish
    
    def forward(self, x):
        """
        Args:
            x: [B, L, D]
        """
        # Apply normalization
        z = self.norm(x)
        
        # Apply input projection
        z = self.in_projection(z)
        
        # Process in chunks for better parallelism
        outs = []
        for i in range(self.num_chunks):
            start_idx = i * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, self.d_model)
            out_chunk = self.S4_parallel[i](z[..., start_idx:end_idx])
            outs.append(out_chunk)
        
        y = torch.cat(outs, dim=-1)
        
        # Apply output projection, activation and dropout
        y = self.out_projection(y)
        y = self.activation(y)
        y = self.dropout(y)
        
        # Residual connection
        return x + y


class ConditionalGenerator(nn.Module):
    """
    Conditional parameter generator for synthesis parameters.
    Takes hidden state and conditioning vector to generate output parameters.
    """
    def __init__(self, hidden_dim, conditioning_dim, output_dim):
        super().__init__()
        
        # Feature-wise linear modulation
        self.conditioning_scale = nn.Linear(conditioning_dim, hidden_dim)
        self.conditioning_shift = nn.Linear(conditioning_dim, hidden_dim)
        
        # Hidden projection (reduce dimension for efficiency)
        reduced_dim = max(32, hidden_dim // 2)
        self.hidden_proj = nn.Sequential(
            nn.Linear(hidden_dim, reduced_dim),
            nn.SiLU()
        )
        
        # Output projection
        self.output_proj = nn.Linear(reduced_dim, output_dim)
    
    def forward(self, hidden, conditioning):
        """
        Args:
            hidden: Sequence of hidden states [batch, seq_len, hidden_dim]
            conditioning: Global conditioning vector [batch, conditioning_dim]
        """
        # Project conditioning vectors
        scale = self.conditioning_scale(conditioning).unsqueeze(1)  # [batch, 1, hidden_dim]
        shift = self.conditioning_shift(conditioning).unsqueeze(1)  # [batch, 1, hidden_dim]
        
        # Apply feature-wise linear modulation
        hidden = hidden * scale + shift
        
        # Project to reduced dimension
        hidden = self.hidden_proj(hidden)
        
        # Generate output parameters
        output = self.output_proj(hidden)
        
        return output


class S4SingVocoder(nn.Module):
    """
    S4-based singing voice synthesis parameter predictor.
    Uses structured state space sequence models for efficient modeling
    of long-range dependencies in the input sequence.
    
    This model is optimized for parameter count and training efficiency
    while maintaining high-quality synthesis capabilities.
    """
    def __init__(
            self,
            phone_map_len, 
            singer_map_len, 
            language_map_len,
            hidden_dim=128, 
            n_harmonics=100,  # Reduced from 150
            n_mag_harmonic=192,  # Reduced from 256
            n_mag_noise=64,  # Reduced from 80
            n_formants=4,
            use_parallel_s4=True):
        super().__init__()
        
        # Embedding dimensions
        self.phone_embed_dim = 64    # Reduced from 128
        self.singer_embed_dim = 32   # Reduced from 64
        self.language_embed_dim = 16 # Reduced from 32
        
        # Embeddings with vocabulary compression
        # For large vocabularies, this helps reduce parameters
        if phone_map_len > 500:
            self.phone_embedding = nn.Sequential(
                nn.Embedding(phone_map_len, 48),
                nn.Linear(48, self.phone_embed_dim)
            )
        else:
            self.phone_embedding = nn.Embedding(phone_map_len, self.phone_embed_dim)
            
        if singer_map_len > 500:
            self.singer_embedding = nn.Sequential(
                nn.Embedding(singer_map_len, 24),
                nn.Linear(24, self.singer_embed_dim)
            )
        else:
            self.singer_embedding = nn.Embedding(singer_map_len, self.singer_embed_dim)
            
        self.language_embedding = nn.Embedding(language_map_len, self.language_embed_dim)
        
        # Calculate combined dimension
        combined_dim = self.phone_embed_dim + self.singer_embed_dim + self.language_embed_dim + 1  # +1 for f0
        
        # Input projection to hidden dimension
        self.input_proj = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )
        
        # S4 layers (efficient sequence modeling)
        self.use_parallel_s4 = use_parallel_s4
        
        S4Class = ParallelS4Layer if use_parallel_s4 else S4Layer
        self.s4_layers = nn.ModuleList([
            S4Class(
                d_model=hidden_dim,
                d_state=64,
                dropout=0.1,
                activation='silu',
                bidirectional=True,
                chunk_size=16 if use_parallel_s4 else None
            )
            for _ in range(3)  # Using just 3 layers for efficiency
        ])
        
        # Global parameter generation (voice characteristics)
        self.global_param_attn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        self.global_params = nn.Linear(hidden_dim, 32)
        
        # Parameter generators with efficient conditioning
        self.param_generators = nn.ModuleDict({
            # Core parameters
            'f0': ConditionalGenerator(hidden_dim, 32, 1),
            'A': ConditionalGenerator(hidden_dim, 32, 1),
            'amplitudes': ConditionalGenerator(hidden_dim, 32, n_harmonics),
            'harmonic_magnitude': ConditionalGenerator(hidden_dim, 32, n_mag_harmonic),
            'noise_magnitude': ConditionalGenerator(hidden_dim, 32, n_mag_noise),
            
            # Formant parameters (combined for efficiency)
            'formants': ConditionalGenerator(hidden_dim, 32, n_formants * 3),  # freq, width, gain
            
            # Expression parameters (combined for efficiency)
            'expression': ConditionalGenerator(hidden_dim, 32, 5)  # vibrato(3) + voice_quality(2)
        })
        
        # Store dimensions for parameter splitting
        self.n_formants = n_formants
        self.n_harmonics = n_harmonics
        self.output_splits = {
            'f0': 1,
            'A': 1,
            'amplitudes': n_harmonics,
            'harmonic_magnitude': n_mag_harmonic,
            'noise_magnitude': n_mag_noise,
            'formant_freqs': n_formants,
            'formant_widths': n_formants,
            'formant_gains': n_formants,
            'vibrato_rate': 1,
            'vibrato_depth': 1, 
            'vibrato_delay': 1,
            'breathiness': 1,
            'tension': 1
        }
    
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
        
        # Apply input projection
        x = self.input_proj(combined)
        
        # Apply S4 layers
        for layer in self.s4_layers:
            x = layer(x)
        
        # Extract global voice characteristics with attention
        attention_weights = self.global_param_attn(x)  # [B, T, 1]
        weighted_features = x * attention_weights
        global_features = weighted_features.sum(dim=1)  # [B, hidden_dim]
        global_params = self.global_params(global_features)  # [B, 32]
        
        # Generate parameters using conditional generators
        raw_params = {}
        for param_name, generator in self.param_generators.items():
            raw_params[param_name] = generator(x, global_params)
        
        # Process and organize the output to match the expected format
        params = {}
        
        # Core parameters
        params['f0'] = torch.sigmoid(raw_params['f0'])
        params['A'] = torch.sigmoid(raw_params['A'])
        
        # Harmonic amplitudes (normalized to distribution)
        amplitudes = F.softmax(raw_params['amplitudes'], dim=-1)
        params['amplitudes'] = amplitudes
        
        # Spectral parameters
        params['harmonic_magnitude'] = torch.sigmoid(raw_params['harmonic_magnitude'])
        params['noise_magnitude'] = torch.sigmoid(raw_params['noise_magnitude'])
        
        # Formant parameters
        formants = raw_params['formants']
        params['formant_freqs'] = torch.sigmoid(formants[..., :self.n_formants]) * 2.0 + 0.5  # Range: 0.5-2.5
        params['formant_widths'] = F.softplus(formants[..., self.n_formants:2*self.n_formants]) + 50.0  # Range: 50-∞
        params['formant_gains'] = torch.tanh(formants[..., 2*self.n_formants:]) * 10.0  # Range: -10 to 10 dB
        
        # Expression parameters
        expression = raw_params['expression']
        params['vibrato_rate'] = torch.sigmoid(expression[..., 0:1]) * 8.0  # Range: 0-8 Hz
        params['vibrato_depth'] = torch.sigmoid(expression[..., 1:2]) * 50.0  # Range: 0-50 cents
        params['vibrato_delay'] = torch.sigmoid(expression[..., 2:3]) * 0.5  # Range: 0-500ms
        params['breathiness'] = torch.sigmoid(expression[..., 3:4])  # Range: 0-1
        params['tension'] = torch.sigmoid(expression[..., 4:5])  # Range: 0-1
        
        return params