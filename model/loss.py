import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np

class HybridLoss(nn.Module):
    """
    A multi-component loss function for neural audio synthesis with specific focus on singing voice.
    Combines spectral losses, F0 (pitch) loss, and harmonic amplitude loss components.
    
    All components can be individually enabled/disabled and weighted to customize the loss function.
    """
    def __init__(
        self,
        # Component enable/disable flags
        use_mel_loss=False,          # Mel-spectrogram loss
        use_mss_loss=True,           # Multi-scale spectral loss
        use_f0_loss=True,            # Fundamental frequency loss
        use_amplitude_loss=True,     # Harmonic amplitude loss
        use_sc_loss=True,            # Spectral convergence loss component within MSS
        
        # FFT configuration
        n_ffts=[1024, 512, 256, 128],  # Multiple FFT sizes for multi-resolution analysis (from config)
        
        # Audio parameters
        sample_rate=24000,           # Sample rate from config
        n_mels=80,                   # Number of mel bands
        
        # Loss component weights
        mel_weight=2.0,              # Weight for mel spectrogram loss (from config)
        mss_weight=1.0,              # Weight for multi-scale spectral loss
        f0_weight=0.1,               # Weight for F0 loss
        amplitude_weight=0.5,        # Weight for amplitude loss (from config)
        sc_weight=0.5,               # Weight for spectral convergence loss within MSS
        
        # F0 configuration
        f0_log_scale=True,           # Use log scale for F0 to better match human pitch perception
        
        # Mel spectrogram configuration
        mel_fmin=40.0,               # Minimum frequency for mel spectrogram
        mel_fmax=12000.0             # Maximum frequency for mel spectrogram (from config)
    ):
        super().__init__()
        # Loss component enable/disable flags
        self.use_mel_loss = use_mel_loss
        self.use_mss_loss = use_mss_loss
        self.use_f0_loss = use_f0_loss
        self.use_amplitude_loss = use_amplitude_loss
        self.use_sc_loss = use_sc_loss
        
        # Store FFT sizes for multi-scale analysis
        self.n_ffts = n_ffts
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        
        # Loss component weights
        self.mel_weight = mel_weight
        self.mss_weight = mss_weight
        self.f0_weight = f0_weight
        self.amplitude_weight = amplitude_weight
        self.sc_weight = sc_weight
        self.f0_log_scale = f0_log_scale
        
        # Mel spectrogram parameters
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        
        # Initialize mel spectrogram transforms for each FFT size
        if self.use_mel_loss:
            self.mel_transforms = nn.ModuleList([
                torchaudio.transforms.MelSpectrogram(
                    sample_rate=sample_rate,
                    n_fft=n_fft,
                    hop_length=n_fft // 4,
                    n_mels=n_mels,
                    f_min=mel_fmin,
                    f_max=mel_fmax,
                    power=1.0  # Use amplitude spectrogram instead of power
                ) for n_fft in n_ffts
            ])
            print(f"Initialized {len(n_ffts)} mel transforms with {n_mels} mel bins each")
        
        # L1 loss for spectral differences
        self.l1_loss = nn.L1Loss()
        
        # For MSS loss (multi-scale spectrogram)
        if self.use_mss_loss:
            self.window_funcs = {
                n_fft: torch.hann_window(n_fft).to(self.device_check()) 
                for n_fft in n_ffts
            }
    
    def device_check(self):
        """Determine device for window functions"""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def compute_stft_loss(self, y_pred, y_true, n_fft):
        """
        Compute STFT (Short-Time Fourier Transform) based loss
        
        Combines log-magnitude loss and spectral convergence loss (optional)
        """
        hop_length = n_fft // 4
        window = self.window_funcs[n_fft]
        
        # Move window to the correct device if needed
        if window.device != y_pred.device:
            window = window.to(y_pred.device)
            self.window_funcs[n_fft] = window
        
        # Compute STFTs
        stft_pred = torch.stft(
            y_pred, 
            n_fft=n_fft, 
            hop_length=hop_length, 
            window=window, 
            return_complex=True
        )
        stft_true = torch.stft(
            y_true, 
            n_fft=n_fft, 
            hop_length=hop_length, 
            window=window, 
            return_complex=True
        )
        
        # Convert to magnitude
        mag_pred = torch.abs(stft_pred)
        mag_true = torch.abs(stft_true)
        
        # Compute log magnitude with stabilization
        log_mag_pred = torch.log(torch.clamp(mag_pred, min=1e-5))
        log_mag_true = torch.log(torch.clamp(mag_true, min=1e-5))
        
        if log_mag_pred.shape[2] > log_mag_true.shape[2]:
            diff = log_mag_pred.shape[2] - log_mag_true.shape[2]
            log_mag_pred = log_mag_pred[:, :, :-diff]

        #print( log_mag_pred.shape )
        #print( log_mag_true.shape )

        # Compute L1 loss on log magnitudes
        loss_mag = F.l1_loss(log_mag_pred, log_mag_true)
        
        # Initialize combined loss with magnitude loss
        loss = loss_mag
        
        if mag_pred.shape[2] > mag_true.shape[2]:
            diff = mag_pred.shape[2] - mag_true.shape[2]
            mag_pred = mag_pred[:, :, :-diff]

        #print( mag_pred.shape )
        #print( mag_true.shape )
        
        # Add spectral convergence loss if enabled
        if self.use_sc_loss:
            sc_loss = torch.norm(mag_true - mag_pred, p='fro') / (torch.norm(mag_true, p='fro') + 1e-7)
            loss = loss_mag + self.sc_weight * sc_loss
        
        return loss
    
    def compute_mel_loss(self, y_pred, y_true, mel_input=None):
        """
        Compute Mel Spectrogram loss, optionally using pre-computed mel_input
        Handles potential dimension mismatches between predicted and target mel spectrograms
        """
        mel_loss = 0.0
        
        # Use pre-computed mel spectrograms if provided
        if mel_input is not None:
            # Assuming mel_input contains the ground truth mel spectrogram
            # We still need to compute the predicted mel spectrogram
            with torch.no_grad():
                # Use the first mel transform as default
                mel_transform = self.mel_transforms[0].to(y_pred.device)
                mel_pred = mel_transform(y_pred)
                
                mel_pred = mel_pred.permute(0, 2, 1)

                if mel_input.shape[1] > mel_pred.shape[1]:
                    diff = mel_input.shape[1] - mel_pred.shape[1]
                    mel_input = mel_input[:, :-diff, :]

                #print( mel_pred.shape )
                #print( mel_input.shape )

                # Apply log with stabilization
                mel_pred = torch.log(torch.clamp(mel_pred, min=1e-5))
                
                # Compute L1 loss
                mel_loss = F.l1_loss(mel_pred, mel_input)

        else:
            # Compute mel spectrograms for both predictions and ground truth
            for mel_transform in self.mel_transforms:
                # Move transform to the correct device
                mel_transform = mel_transform.to(y_pred.device)
                
                # Compute spectrograms
                mel_pred = mel_transform(y_pred)
                mel_true = mel_transform(y_true)
                
                # Apply log with stabilization
                log_mel_pred = torch.log(torch.clamp(mel_pred, min=1e-5))
                log_mel_true = torch.log(torch.clamp(mel_true, min=1e-5))
                
                # Compute L1 loss and add to total
                mel_loss += F.l1_loss(log_mel_pred, log_mel_true)
            
            # Average across all transforms
            mel_loss /= len(self.mel_transforms)
            
        return mel_loss
    
    def compute_f0_loss(self, f0_pred, f0_true):
        """
        Compute F0 (fundamental frequency) loss
        """
        # Create mask for valid F0 values (ignore padded or unvoiced frames)
        # Assuming 0 or negative values represent unvoiced or padding
        mask = (f0_true > 0).float()
        
        f0_pred = f0_pred.squeeze(-1)

        #print( f0_pred.shape )
        #print( f0_true.shape )

        # Apply mask to both predictions and ground truth
        f0_pred_masked = f0_pred * mask
        f0_true_masked = f0_true * mask
        
        if self.f0_log_scale:
            # Add small epsilon to avoid log(0)
            epsilon = 1e-8
            
            # Take log of F0 values (better for perceptual differences)
            f0_pred_log = torch.log(f0_pred_masked + epsilon)
            f0_true_log = torch.log(f0_true_masked + epsilon)
            
            # Compute loss on log scale
            loss = F.l1_loss(f0_pred_log, f0_true_log, reduction='sum') / (mask.sum() + epsilon)
        else:
            # Direct L1 loss on Hz values
            loss = F.l1_loss(f0_pred_masked, f0_true_masked, reduction='sum') / (mask.sum() + 1e-8)
        
        return loss
    
    def compute_amplitude_loss(self, amplitudes_pred, amplitudes_true):
        """
        Compute loss on harmonic amplitudes
        """
        if amplitudes_pred is None or amplitudes_true is None:
            return 0.0
            
        # Compute L1 loss on amplitude values
        loss = F.l1_loss(amplitudes_pred, amplitudes_true)
        
        return loss
    
    def forward(self, signal, audio, f0_pred=None, f0_true=None, mel_input=None, 
                amplitudes_pred=None, amplitudes_true=None):
        """
        Compute the combined loss function
        
        Args:
            signal: Predicted audio signal [B, T]
            audio: Ground truth audio signal [B, T]
            f0_pred: Predicted F0 values [B, T']
            f0_true: Ground truth F0 values [B, T']
            mel_input: Optional pre-computed mel spectrogram [B, n_mels, T'']
            amplitudes_pred: Predicted harmonic amplitudes [B, n_harmonics, T']
            amplitudes_true: Ground truth harmonic amplitudes [B, n_harmonics, T']
            
        Returns:
            Dict containing total loss and individual loss components
        """
        # Initialize loss dict
        loss_dict = {}
        total_loss = 0.0
        
        # Track which losses are active in this forward pass
        active_losses = []
        
        # Multi-scale spectral loss
        if self.use_mss_loss:
            mss_loss = 0.0
            for n_fft in self.n_ffts:
                mss_loss += self.compute_stft_loss(signal, audio, n_fft)
            mss_loss /= len(self.n_ffts)  # Average across FFT sizes
            loss_dict['mss_loss'] = self.mss_weight * mss_loss
            total_loss += self.mss_weight * mss_loss
            active_losses.append(f"MSS(w={self.mss_weight:.2f})")
        
        # Mel spectrogram loss
        if self.use_mel_loss:
            mel_loss = self.compute_mel_loss(signal, audio, mel_input)
            loss_dict['mel_loss'] = self.mel_weight * mel_loss 
            total_loss += self.mel_weight * mel_loss 
            active_losses.append(f"Mel(w={self.mel_weight:.2f})")
        
        # F0 loss
        if self.use_f0_loss and f0_pred is not None and f0_true is not None:
            f0_loss = self.compute_f0_loss(f0_pred, f0_true)
            loss_dict['f0_loss'] = self.f0_weight * f0_loss
            total_loss += self.f0_weight * f0_loss
            active_losses.append(f"F0(w={self.f0_weight:.2f})")
        
        # Amplitude loss
        if self.use_amplitude_loss and amplitudes_pred is not None and amplitudes_true is not None:
            amp_loss = self.compute_amplitude_loss(amplitudes_pred, amplitudes_true)
            loss_dict['amplitude_loss'] = self.amplitude_weight * amp_loss
            total_loss += self.amplitude_weight * amp_loss
            active_losses.append(f"Amp(w={self.amplitude_weight:.2f})")
        
        # Add total loss to dict
        loss_dict['loss'] = total_loss
        
        # Add active losses info for logging
        loss_dict['active_losses'] = '+'.join(active_losses)
        
        return loss_dict