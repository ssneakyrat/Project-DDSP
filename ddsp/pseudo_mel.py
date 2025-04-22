# pseudo_mel.py - New component for pseudo-mel generation and formant prediction

import torch
import torch.nn as nn
from torch.nn import functional as F

from ddsp.pcmer import PCmer

class PseudoMelGenerator(nn.Module):
    """
    Compact PCmer-based pseudo-mel generator.
    This serves as an intermediate representation between linguistic features and synthesis parameters.
    """
    def __init__(self, 
                 num_phonemes,
                 num_singers,
                 num_languages,
                 hidden_dim=64,
                 num_layers=2,
                 num_heads=4,
                 mel_dims=80):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mel_dims = mel_dims
        
        # Embeddings for discrete inputs
        self.phoneme_embedding = nn.Embedding(num_phonemes, hidden_dim)
        self.singer_embedding = nn.Embedding(num_singers, hidden_dim)
        self.language_embedding = nn.Embedding(num_languages, hidden_dim)
        
        # Projections for continuous inputs
        self.f0_projection = nn.Linear(1, hidden_dim)
        self.duration_projection = nn.Linear(1, hidden_dim)
        
        # Feature fusion
        self.fusion_layer = nn.Linear(hidden_dim * 5, hidden_dim)
        
        # PCmer for temporal modeling
        self.pcmer = PCmer(
            num_layers=num_layers,
            num_heads=num_heads,
            dim_model=hidden_dim,
            dim_keys=hidden_dim,
            dim_values=hidden_dim,
            residual_dropout=0.1,
            attention_dropout=0.1)
        
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Project to pseudo-mel dimensions
        self.to_mel = nn.Linear(hidden_dim, mel_dims)
        
    def forward(self, phonemes, durations, f0, singer_ids, language_ids):
        """
        Forward pass of the pseudo-mel generator.
        
        Args:
            phonemes: Phoneme sequence [B, T_phones]
            durations: Phoneme durations [B, T_phones]
            f0: Fundamental frequency [B, T_frames]
            singer_ids: Singer IDs [B]
            language_ids: Language IDs [B]
            
        Returns:
            pseudo_mel: Pseudo-mel representation [B, T_frames, mel_dims]
            hidden_states: Hidden states for skip connections [B, T_frames, hidden_dim]
        """
        # Get embeddings
        phone_emb = self.phoneme_embedding(phonemes)
        singer_emb = self.singer_embedding(singer_ids).unsqueeze(1).expand(-1, phonemes.size(1), -1)
        lang_emb = self.language_embedding(language_ids).unsqueeze(1).expand(-1, phonemes.size(1), -1)        

        # Process continuous features
        f0_emb = self.f0_projection(f0.unsqueeze(-1))
        dur_emb = self.duration_projection(durations.unsqueeze(-1))
        
        print("Phoneme embedding shape:", phone_emb.size())
        print("f0 embedding shape:", f0_emb.size())
        print('mel duration:', dur_emb.size())
        print("Singer embedding shape:", singer_emb.size())
        print("Language embedding shape:", lang_emb.size())

        # Combine all embeddings
        combined = torch.cat([phone_emb, singer_emb, lang_emb, f0_emb], dim=-1)
        fused = self.fusion_layer(combined)
        
        # Temporal modeling with PCmer
        hidden_states = self.pcmer(fused)
        hidden_states = self.norm(hidden_states)
        
        # Project to pseudo-mel dimensions
        pseudo_mel = self.to_mel(hidden_states)
        
        return pseudo_mel, hidden_states

class FormantParameterPredictor(nn.Module):
    """
    Formant parameter predictor using a compact PCmer architecture.
    Takes both pseudo-mel and phonetic features via skip connections.
    """
    def __init__(self,
                 num_phonemes,
                 hidden_dim=32,
                 num_layers=1,
                 num_heads=2,
                 n_formants=5,
                 mel_dims=80):
        super().__init__()
        self.n_formants = n_formants
        
        # Direct phoneme conditioning
        self.phoneme_embedding = nn.Embedding(num_phonemes, hidden_dim)
        
        # Input projections
        self.mel_projection = nn.Linear(mel_dims, hidden_dim)
        self.hidden_projection = nn.Linear(64, hidden_dim)  # From pseudo-mel generator hidden states
        
        # PCmer for temporal modeling
        self.pcmer = PCmer(
            num_layers=num_layers,
            num_heads=num_heads,
            dim_model=hidden_dim,
            dim_keys=hidden_dim,
            dim_values=hidden_dim,
            residual_dropout=0.1,
            attention_dropout=0.1)
        
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Output projections for formant parameters
        self.formant_freq_out = nn.Linear(hidden_dim, n_formants)
        self.formant_bw_out = nn.Linear(hidden_dim, n_formants)
        self.formant_amp_out = nn.Linear(hidden_dim, n_formants)
        
    def forward(self, pseudo_mel, hidden_states, phonemes):
        """
        Forward pass of the formant parameter predictor.
        
        Args:
            pseudo_mel: Pseudo-mel representation [B, T_frames, mel_dims]
            hidden_states: Hidden states from pseudo-mel generator [B, T_frames, hidden_dim_generator]
            phonemes: Phoneme sequence [B, T_phones]
            
        Returns:
            formant_parameters: Tuple of (frequencies, bandwidths, amplitudes)
        """
        # Process pseudo-mel
        mel_features = self.mel_projection(pseudo_mel)
        
        # Process hidden states from pseudo-mel generator
        hidden_features = self.hidden_projection(hidden_states)
        
        # Get phoneme embeddings and resize to match frame length
        phone_emb = self.phoneme_embedding(phonemes)
        if phone_emb.size(1) != pseudo_mel.size(1):
            phone_emb = F.interpolate(
                phone_emb.transpose(1, 2),  # [B, hidden_dim, T_phones]
                size=pseudo_mel.size(1),    # Target length = frame length
                mode='nearest'
            ).transpose(1, 2)  # [B, T_frames, hidden_dim]
        
        # Combine features with skip connections
        combined = mel_features + hidden_features + phone_emb
        
        # Process with PCmer
        x = self.pcmer(combined)
        x = self.norm(x)
        
        # Generate formant parameters
        formant_freqs = self.formant_freq_out(x)
        formant_bws = self.formant_bw_out(x)
        formant_amps = self.formant_amp_out(x)
        
        # Apply activations for appropriate ranges
        formant_freqs = torch.sigmoid(formant_freqs) * 5000.0 + 100.0  # Range 100-5100Hz
        formant_bws = torch.sigmoid(formant_bws) * 500.0 + 50.0        # Range 50-550Hz
        formant_amps = F.softmax(formant_amps, dim=-1)                  # Normalized amplitudes
        
        return (formant_freqs, formant_bws, formant_amps)
