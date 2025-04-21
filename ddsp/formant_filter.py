import torch
import torch.nn as nn
import numpy as np
from ddsp.core import upsample

# Import the patched core functions
from ddsp.core import fft_convolve, apply_window_to_impulse_response

def frequency_impulse_response(magnitudes, window_size: int = 0):
    """Get windowed impulse responses using the frequency sampling method.
    Patched to fix device issues.
    """
    # Get the IR (zero-phase form).
    magnitudes = torch.complex(magnitudes, torch.zeros_like(magnitudes))
    impulse_response = torch.fft.irfft(magnitudes)
    
    # Window and put in causal form.
    impulse_response = apply_window_to_impulse_response(impulse_response, impulse_response.size(-1))
    
    return impulse_response

def frequency_filter(audio, magnitudes, window_size=0, padding='same', mel_scale_noise=False, mel_basis=None):
    """Filter audio with a finite impulse response filter.
    Patched to fix device issues.
    """
    impulse_response = frequency_impulse_response(magnitudes, window_size=window_size)
    
    return fft_convolve(audio, impulse_response, padding=padding, mel_scale_noise=mel_scale_noise)

class FormantFilter(nn.Module):
    """
    Formant filter for vocal tract modeling.
    Patched version that handles device consistency.
    """
    def __init__(self, sampling_rate):
        super().__init__()
        self.sampling_rate = sampling_rate
        
    def forward(self, audio, formant_freqs, formant_bws, formant_amps):
        """
        Apply formant filtering to audio signal
        
        Args:
            audio: Batch of audio signals [B, T]
            formant_freqs: Formant frequencies [B, n_frames, n_formants]
            formant_bws: Formant bandwidths [B, n_frames, n_formants]
            formant_amps: Formant amplitudes [B, n_frames, n_formants]
            
        Returns:
            Filtered audio [B, T]
        """
        # Get device from input
        device = audio.device
        
        # Upsample formant parameters to audio rate
        f_freqs = upsample(formant_freqs, audio.shape[1] // formant_freqs.shape[1])
        f_bws = upsample(formant_bws, audio.shape[1] // formant_bws.shape[1])
        f_amps = upsample(formant_amps, audio.shape[1] // formant_amps.shape[1])
        
        # Create spectral envelope from formant parameters
        n_freqs = 512  # Number of frequency bins for filter
        envelope = torch.zeros(audio.shape[0], f_freqs.shape[1], n_freqs, device=device)
        
        # Create frequency axis scaled to Nyquist
        freq_axis = torch.linspace(0, self.sampling_rate / 2, n_freqs, device=device)
        
        # For each formant, add its contribution to the envelope
        for i in range(formant_freqs.shape[-1]):
            # Extract parameters for this formant
            f0 = f_freqs[..., i].unsqueeze(-1)  # [B, T, 1]
            bw = f_bws[..., i].unsqueeze(-1)    # [B, T, 1]
            amp = f_amps[..., i].unsqueeze(-1)  # [B, T, 1]
            
            # Calculate formant contribution (resonance curve)
            # Using simplified second-order resonance formula
            response = amp * (bw**2) / ((freq_axis - f0)**2 + bw**2)
            envelope += response
        
        # Apply the spectral envelope using frequency_filter
        filtered_audio = frequency_filter(audio, envelope)
        
        return filtered_audio

def apply_formant_filter_batch(audio, formant_params, sampling_rate):
    """
    Helper function to apply formant filtering in batch mode
    
    Args:
        audio: Audio signals [B, T]
        formant_params: Tuple of (frequencies, bandwidths, amplitudes)
        sampling_rate: Sampling rate in Hz
        
    Returns:
        Filtered audio [B, T]
    """
    formant_filter = FormantFilter(sampling_rate)
    formant_freqs, formant_bws, formant_amps = formant_params
    return formant_filter(audio, formant_freqs, formant_bws, formant_amps)