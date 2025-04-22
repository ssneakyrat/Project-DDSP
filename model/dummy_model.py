import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderBlock(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_size)
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, input_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = x + residual
        return x

class DummyModel(nn.Module):
    def __init__(self, hidden_size, n_layers, dropout, n_mels, 
                 phone_map_len, singer_map_len, language_map_len):
        super().__init__()
        
        # Embedding layers
        self.phone_embedding = nn.Embedding(phone_map_len, hidden_size)
        self.singer_embedding = nn.Embedding(singer_map_len, hidden_size // 4)
        self.language_embedding = nn.Embedding(language_map_len, hidden_size // 4)
        
        # Mel encoder
        self.mel_encoder = nn.Sequential(
            nn.Linear(n_mels, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )
        
        # F0 encoder
        self.f0_encoder = nn.Sequential(
            nn.Linear(1, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU()
        )
        
        # Encoder layers
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(hidden_size, hidden_size * 2, dropout)
            for _ in range(n_layers)
        ])
        
        # Decoder (mel reconstruction)
        self.mel_decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_mels)
        )
    
    def forward(self, mel, phone_seq, f0, singer_id, language_id):
        batch_size, seq_len, n_mels = mel.shape
        
        # Process phone sequence
        phone_emb = self.phone_embedding(phone_seq)  # [B, T]
        
        # Process singer and language IDs
        singer_emb = self.singer_embedding(singer_id)  # [B, 1]
        language_emb = self.language_embedding(language_id)  # [B, 1]
        
        # Expand singer and language embeddings to match sequence length
        singer_emb = singer_emb.expand(-1, seq_len, -1)
        language_emb = language_emb.expand(-1, seq_len, -1)
        
        # Process mel spectrogram
        mel_enc = self.mel_encoder(mel)
        
        # Process F0
        f0_reshaped = f0.unsqueeze(-1)  # Add feature dimension
        f0_enc = self.f0_encoder(f0_reshaped)
        
        # Combine features
        combined = mel_enc #+ phone_emb
        
        # Apply encoder blocks
        for block in self.encoder_blocks:
            combined = block(combined)
        
        # Decode mel spectrogram
        mel_output = self.mel_decoder(combined)
        
        return {
            'mel_output': mel_output,
            'encoded_features': combined
        }