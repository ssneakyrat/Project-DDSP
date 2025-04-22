import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from ddsp.core import remove_above_nyquist

class FormantHarmonicOscillator(nn.Module):
    """
    Integrated harmonic oscillator with formant filtering.
    
    This class combines harmonic synthesis and formant filtering into a single
    operation for improved efficiency by modulating harmonic amplitudes directly
    instead of post-filtering.
    """
    def __init__(self, sampling_rate, n_harmonics=100, oscillator=torch.sin):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.n_harmonics = n_harmonics
        self.oscillator = oscillator
        
    def forward(self, f0, amplitudes, formant_freqs, formant_bws, formant_amps, initial_phase=None):
        """
        Generate harmonic signal with built-in formant characteristics.
        
        Args:
            f0: Fundamental frequency [B, T, 1] (Hz)
            amplitudes: Harmonic amplitudes [B, T, n_harmonic]
            formant_freqs: Formant frequencies [B, C, n_formants] (Hz)
            formant_bws: Formant bandwidths [B, C, n_formants] (Hz)
            formant_amps: Formant amplitudes [B, C, n_formants] (normalized)
            initial_phase: Optional initial phase [B, 1, 1]
            
        Returns:
            signal: Output audio signal [B, T]
            final_phase: Final phase state [B, 1, 1]
        """
        batch_size, time_steps, _ = f0.shape
        device = f0.device
        n_base_harmonics = amplitudes.shape[-1]
        n_formants = formant_freqs.shape[-1]
        
        # Default phase if not provided
        if initial_phase is None:
            initial_phase = torch.zeros(batch_size, 1, 1, device=device)
        
        # Silence mask and detach f0 for stability
        mask = (f0 > 0).detach()
        f0 = f0.detach()
        
        # Compute phase progression for harmonics
        phase = torch.cumsum(2 * np.pi * f0 / self.sampling_rate, dim=1) + initial_phase
        
        # Check if formant parameters need upsampling
        if formant_freqs.shape[1] != time_steps:
            # Interpolate formant parameters to audio time steps using F.interpolate
            # For each formant parameter, reshape for interpolation and reshape back
            formant_freqs_upsampled = F.interpolate(
                formant_freqs.transpose(1, 2),  # [B, n_formants, control_steps]
                size=time_steps,
                mode='linear',
                align_corners=True
            ).transpose(1, 2)  # [B, time_steps, n_formants]
            
            formant_bws_upsampled = F.interpolate(
                formant_bws.transpose(1, 2),
                size=time_steps, 
                mode='linear',
                align_corners=True
            ).transpose(1, 2)
            
            formant_amps_upsampled = F.interpolate(
                formant_amps.transpose(1, 2),
                size=time_steps,
                mode='linear',
                align_corners=True
            ).transpose(1, 2)
        else:
            formant_freqs_upsampled = formant_freqs
            formant_bws_upsampled = formant_bws
            formant_amps_upsampled = formant_amps
        
        # Create output signal tensor
        signal = torch.zeros(batch_size, time_steps, device=device)
        
        # Process each harmonic - we'll limit to max(n_base_harmonics, 60) for efficiency
        max_harmonics = min(self.n_harmonics, 60)
        
        for h in range(1, max_harmonics + 1):
            # Get current harmonic frequency
            h_freq = h * f0  # [B, T, 1]
            
            # Skip harmonics above Nyquist
            nyquist_mask = (h_freq < self.sampling_rate/2).squeeze(-1)
            if not nyquist_mask.any():
                continue
                
            # Get base harmonic amplitude (reuse amplitudes for higher harmonics)
            h_idx = min(h-1, n_base_harmonics-1)
            base_amplitude = amplitudes[:, :, h_idx]
            
            # Calculate formant influence on this harmonic
            # Initialize with ones (neutral effect)
            formant_factor = torch.ones_like(base_amplitude)
            
            # For each formant, apply resonance effects
            for f in range(n_formants):
                freq = formant_freqs_upsampled[:, :, f].unsqueeze(-1)  # [B, T, 1]
                bw = formant_bws_upsampled[:, :, f].unsqueeze(-1)      # [B, T, 1]
                amp = formant_amps_upsampled[:, :, f].unsqueeze(-1)    # [B, T, 1]
                
                # Apply formant resonance using second-order resonator gain formula
                # This simulates the frequency response of a biquad resonator
                dist_squared = (h_freq - freq)**2
                resonance = amp * (bw**2) / (dist_squared + bw**2)
                
                # Accumulate resonance effects (multiplicative for proper filter shape)
                formant_factor = formant_factor * (1.0 + resonance.squeeze(-1))
            
            # Compute harmonic with modulated amplitude
            h_phase = phase * h
            h_signal = self.oscillator(h_phase).squeeze(-1) * base_amplitude * formant_factor
            
            # Apply Nyquist and silence masks
            h_signal = h_signal * nyquist_mask * mask.squeeze(-1)
            
            # Add to output
            signal += h_signal
        
        # Return output and final phase
        final_phase = phase[:, -1:, :] % (2 * np.pi)
        return signal, final_phase.detach()