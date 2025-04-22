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
                 mel_dims=80):
        super().__init__()
        self.mel_dims = mel_dims
        
        # Embeddings for discrete inputs
        self.phoneme_embedding = nn.Embedding(num_phonemes, mel_dims)
        self.singer_embedding = nn.Embedding(num_singers, mel_dims)
        self.language_embedding = nn.Embedding(num_languages, mel_dims)
        
        # Projections for continuous inputs
        self.f0_projection = nn.Linear(1, mel_dims)
        
        # Feature fusion
        self.fusion_layer = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1),  # 4 input channels for the 4 embeddings
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=3, padding=1)    # Output a single channel
        )
        
    def forward(self, phonemes, f0, singer_ids, language_ids):
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
        
        print("Phoneme shape:", phonemes.size())
        print("f0 shape:", f0.size())
        print("Singer shape:", singer_ids.size())
        print("Language shape:", language_ids.size())

        print("Phoneme embedding shape:", phone_emb.size())
        print("f0 embedding shape:", f0_emb.size())
        print("Singer embedding shape:", singer_emb.size())
        print("Language embedding shape:", lang_emb.size())

        # Stack all embeddings as channels for the fusion layer
        embeddings = torch.stack([phone_emb, f0_emb, singer_emb, lang_emb], dim=1)  # [16, 4, 201, 80]

        # Apply fusion layer
        fused = self.fusion_layer(embeddings)  # [16, 1, 201, 80]
        pseudo_mel = fused.squeeze(1)  # [16, 201, 80]
        
        print("pseudo_mel fused shape:", pseudo_mel.size())

        return pseudo_mel

class FormantParameterPredictor(nn.Module):
    """
    Formant parameter predictor using a compact PCmer architecture.
    Takes both pseudo-mel and phonetic features via skip connections.
    """
    def __init__(self, input_channel,
                 hidden_dim=32,
                 n_formants=5):
        super().__init__()
        self.n_formants = n_formants
        
        # conv in stack
        self.stack = nn.Sequential(
                nn.Conv1d(input_channel, hidden_dim, 3, 1, 1),
                nn.GroupNorm(4, hidden_dim),
                nn.LeakyReLU(),
                nn.Conv1d(hidden_dim, hidden_dim, 3, 1, 1)) 
        
        # PCmer for temporal modeling
        self.pcmer = PCmer(
            num_layers=3,
            num_heads=8,
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
        
    def forward(self, pseudo_mel):
        """
        Forward pass of the formant parameter predictor.
        
        Args:
            pseudo_mel: Pseudo-mel representation [B, T_frames, mel_dims]
            hidden_states: Hidden states from pseudo-mel generator [B, T_frames, hidden_dim_generator]
            phonemes: Phoneme sequence [B, T_phones]
            
        Returns:
            formant_parameters: Tuple of (frequencies, bandwidths, amplitudes)
        """
        pseudo_mel = self.stack(pseudo_mel.transpose(1,2)).transpose(1,2)
        # Process with PCmer
        x = self.pcmer(pseudo_mel)
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
