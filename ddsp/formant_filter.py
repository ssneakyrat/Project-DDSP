import torch
import torch.nn as nn
import numpy as np
from ddsp.core import upsample, frequency_filter

class FormantFilter(nn.Module):
    """
    Formant filter for vocal tract modeling.
    
    Implements a differentiable filter that can shape harmonic content
    according to formant parameters (frequencies, bandwidths, amplitudes).
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
        # Upsample formant parameters to audio rate
        f_freqs = upsample(formant_freqs, audio.shape[1] // formant_freqs.shape[1])
        f_bws = upsample(formant_bws, audio.shape[1] // formant_bws.shape[1])
        f_amps = upsample(formant_amps, audio.shape[1] // formant_amps.shape[1])
        
        # Create spectral envelope from formant parameters
        n_freqs = 512  # Number of frequency bins for filter
        envelope = torch.zeros(audio.shape[0], f_freqs.shape[1], n_freqs, device=audio.device)
        
        # Create frequency axis scaled to Nyquist
        freq_axis = torch.linspace(0, self.sampling_rate / 2, n_freqs, device=audio.device)
        
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