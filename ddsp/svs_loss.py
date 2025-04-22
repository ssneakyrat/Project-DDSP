import torch
import torch.nn as nn
from torch.nn import functional as F
import torchaudio
import numpy as np

from ddsp.loss import MSSLoss, F0L1Loss

class MelSpectrogramLoss(nn.Module):
    """
    Mel-spectrogram loss for comparing spectral characteristics
    """
    def __init__(self, sample_rate, n_fft=1024, hop_length=256, n_mels=80):
        super().__init__()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=20.0,
            f_max=sample_rate/2.0
        )
        
    def forward(self, y_pred, y_true):
        # Ensure both inputs have the same length
        min_length = min(y_pred.shape[-1], y_true.shape[-1])
        y_pred = y_pred[..., :min_length]
        y_true = y_true[..., :min_length]
        
        # Calculate power mel spectrograms
        if len(y_pred.shape) == 3:
            y_pred = y_pred.squeeze(1)
        if len(y_true.shape) == 3:
            y_true = y_true.squeeze(1)
            
        mel_pred = self.mel_transform(y_pred)
        mel_true = self.mel_transform(y_true)
        
        # Convert to log scale
        log_mel_pred = torch.log(mel_pred + 1e-5)
        log_mel_true = torch.log(mel_true + 1e-5)
        
        # L1 loss
        loss = F.l1_loss(log_mel_pred, log_mel_true)
        
        return loss

class SVSHybridLoss(nn.Module):
    """
    Combined loss for singing voice synthesis
    
    Combines multi-scale spectral loss, F0 loss, and mel-spectrogram loss
    """
    def __init__(self, n_ffts, sample_rate):
        super().__init__()
        self.loss_mss_func = MSSLoss(n_ffts)
        #self.f0_loss_func = F0L1Loss()
        self.mel_loss_func = MelSpectrogramLoss(sample_rate)
        
    def forward(self, y_pred, y_true, f0_pred, f0_true):
        """
        Compute combined loss
        
        Args:
            y_pred: Predicted audio
            y_true: Ground truth audio
            f0_pred: Predicted F0
            f0_true: Ground truth F0
            
        Returns:
            total_loss: Combined loss value
            individual_losses: Tuple of (mss_loss, f0_loss, mel_loss)
        """
        # Original losses
        loss_mss = self.loss_mss_func(y_pred, y_true)
        #loss_f0 = self.f0_loss_func(f0_pred, f0_true)
        
        # New mel-spectrogram loss
        loss_mel = self.mel_loss_func(y_pred, y_true)
        
        # Combined loss
        loss = loss_mss + loss_mel
        
        return loss, (loss_mss, loss_mel)