import torch
import torch.nn as nn
from torch.nn import functional as F

from ddsp.pcmer import PCmer

def split_to_dict(tensor, tensor_splits):
    """Split a tensor into a dictionary of multiple tensors."""
    labels = []
    sizes = []

    for k, v in tensor_splits.items():
        labels.append(k)
        sizes.append(v)

    tensors = torch.split(tensor, sizes, dim=-1)
    return dict(zip(labels, tensors))

class Phoneme2Control(nn.Module):
    def __init__(self, num_phonemes, num_singers, num_languages, output_splits, hidden_dim=256):
        super().__init__()
        self.output_splits = output_splits
        
        # Embeddings for discrete inputs
        self.phoneme_embedding = nn.Embedding(num_phonemes, hidden_dim)
        self.singer_embedding = nn.Embedding(num_singers, hidden_dim)
        self.language_embedding = nn.Embedding(num_languages, hidden_dim)
        
        # Projections for continuous inputs
        self.f0_projection = nn.Linear(1, hidden_dim)
        self.duration_projection = nn.Linear(1, hidden_dim)
        
        # Feature fusion
        self.fusion_layer = nn.Linear(hidden_dim * 5, hidden_dim)
        
        # Temporal modeling with transformer
        self.transformer = PCmer(
            num_layers=6,
            num_heads=8,
            dim_model=hidden_dim,
            dim_keys=hidden_dim,
            dim_values=hidden_dim,
            residual_dropout=0.1,
            attention_dropout=0.1)
        
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Output projections
        self.n_out = sum([v for k, v in output_splits.items()])
        self.dense_out = nn.Linear(hidden_dim, self.n_out)
        
        # Formant parameter prediction
        self.n_formants = 5
        self.formant_freq_out = nn.Linear(hidden_dim, self.n_formants)
        self.formant_bw_out = nn.Linear(hidden_dim, self.n_formants)
        self.formant_amp_out = nn.Linear(hidden_dim, self.n_formants)
        
    def forward(self, phonemes, durations, f0, singer_ids, language_ids):
        # Get embeddings
        phone_emb = self.phoneme_embedding(phonemes)
        singer_emb = self.singer_embedding(singer_ids).unsqueeze(1).expand(-1, phonemes.size(1), -1)
        lang_emb = self.language_embedding(language_ids).unsqueeze(1).expand(-1, phonemes.size(1), -1)
        
        # Process continuous features
        f0_emb = self.f0_projection(f0.unsqueeze(-1))
        dur_emb = self.duration_projection(durations.unsqueeze(-1))
        
        # Combine features
        combined = torch.cat([phone_emb, singer_emb, lang_emb, f0_emb, dur_emb], dim=-1)
        fused = self.fusion_layer(combined)
        
        # Temporal modeling
        x = self.transformer(fused)
        x = self.norm(x)
        
        # Generate control parameters
        e = self.dense_out(x)
        controls = split_to_dict(e, self.output_splits)
        
        # Generate formant parameters
        formant_freqs = self.formant_freq_out(x)
        formant_bws = self.formant_bw_out(x)
        formant_amps = self.formant_amp_out(x)
        
        # Activate with appropriate ranges
        formant_freqs = torch.sigmoid(formant_freqs) * 5000.0 + 100.0  # Range 100-5100Hz
        formant_bws = torch.sigmoid(formant_bws) * 500.0 + 50.0        # Range 50-550Hz
        formant_amps = torch.nn.functional.softmax(formant_amps, dim=-1)  # Normalized amplitudes
        
        return controls, (formant_freqs, formant_bws, formant_amps)