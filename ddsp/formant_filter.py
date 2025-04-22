import torch
import torch.nn as nn
import math

class BiquadResonator(nn.Module):
    """
    Single biquad resonator for a formant.
    
    Implements a time-domain second-order IIR filter configured 
    as a resonator for a single formant.
    """
    def __init__(self, sampling_rate):
        super().__init__()
        self.sampling_rate = sampling_rate
    
    def forward(self, x, freq, bw, gain, initial_state=None):
        """
        Apply biquad resonator to input signal.
        
        Args:
            x: Input audio [B, T]
            freq: Formant frequency in Hz [B, C] where C is control rate
            bw: Formant bandwidth in Hz [B, C]
            gain: Formant amplitude [B, C]
            initial_state: Optional tuple of (y1, y2, x1, x2) from previous block
            
        Returns:
            Tuple of (output, final_state)
        """
        batch_size, block_size = x.shape
        device = x.device
        
        # Normalize frequency and bandwidth to (0, 1)
        nyquist = self.sampling_rate / 2
        freq_norm = freq / nyquist
        bw_norm = bw / nyquist
        
        # Clamp to prevent instability
        freq_norm = torch.clamp(freq_norm, 1e-5, 0.95)
        bw_norm = torch.clamp(bw_norm, 1e-5, 0.95)
        
        # Get initial state
        if initial_state is None:
            y1 = torch.zeros(batch_size, device=device)
            y2 = torch.zeros(batch_size, device=device)
            x1 = torch.zeros(batch_size, device=device)
            x2 = torch.zeros(batch_size, device=device)
        else:
            y1, y2, x1, x2 = initial_state
        
        # Output array
        output = torch.zeros_like(x)
        
        # Get number of control points and calculate time scaling
        n_controls = freq.shape[1]
        time_scale = block_size / n_controls
        
        # Process sample-by-sample (we keep this for clarity and stability)
        for t in range(block_size):
            # Calculate control frame index (interpolating between control points)
            cf = t / time_scale  # Continuous index into control frames
            cf_idx = int(cf)  # Integer index
            cf_frac = cf - cf_idx  # Fractional part for interpolation
            cf_next = min(cf_idx + 1, n_controls - 1)  # Next control frame index
            
            # Interpolate parameters
            freq_t = freq_norm[:, cf_idx] * (1 - cf_frac) + freq_norm[:, cf_next] * cf_frac
            bw_t = bw_norm[:, cf_idx] * (1 - cf_frac) + bw_norm[:, cf_next] * cf_frac
            gain_t = gain[:, cf_idx] * (1 - cf_frac) + gain[:, cf_next] * cf_frac
            
            # Calculate biquad coefficients
            omega = 2 * math.pi * freq_t
            sin_omega = torch.sin(omega)
            cos_omega = torch.cos(omega)
            # Use math.log(2) since it's a constant scalar
            log2_val = math.log(2)
            alpha = sin_omega * torch.sinh(log2_val / 2 * bw_t * omega / sin_omega)
            
            # Calculate coefficients
            b0 = alpha * gain_t
            b1 = torch.zeros_like(b0)
            b2 = -alpha * gain_t
            a0 = 1 + alpha
            a1 = -2 * cos_omega
            a2 = 1 - alpha
            
            # Normalize by a0
            b0 = b0 / a0
            b1 = b1 / a0
            b2 = b2 / a0
            a1 = a1 / a0
            a2 = a2 / a0
            
            # Direct Form I implementation
            x_t = x[:, t]
            y_t = b0 * x_t + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            
            # Update state
            y2, y1 = y1, y_t
            x2, x1 = x1, x_t
            
            # Store output
            output[:, t] = y_t
        
        # Return output and final state
        final_state = (y1, y2, x1, x2)
        return output, final_state


class FormantFilter(nn.Module):
    """
    Memory-efficient formant filter using parallel biquad resonators.
    
    Processes audio in blocks to reduce memory usage, particularly
    suitable for singing synthesis where smooth parameter transitions
    are critical.
    """
    def __init__(self, sampling_rate, n_formants=5, block_size=128):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.n_formants = n_formants
        self.block_size = block_size
        
        # Create resonators for each formant
        self.resonators = nn.ModuleList([
            BiquadResonator(sampling_rate) for _ in range(n_formants)
        ])
    
    def forward(self, audio, formant_freqs, formant_bws, formant_amps):
        """
        Apply formant filtering with memory-efficient block processing.
        
        Args:
            audio: Audio signal [B, T]
            formant_freqs: Formant frequencies in Hz [B, C, F] where:
                           B = batch size, C = control frames, F = number of formants
            formant_bws: Formant bandwidths in Hz [B, C, F]
            formant_amps: Formant amplitudes [B, C, F]
            
        Returns:
            Filtered audio [B, T]
        """
        batch_size, audio_len = audio.shape
        device = audio.device
        
        # Initialize output tensor
        output = torch.zeros_like(audio)
        
        # Initialize resonator states
        states = [None] * self.n_formants
        
        # Process audio in blocks
        for block_start in range(0, audio_len, self.block_size):
            # Calculate block boundaries
            block_end = min(block_start + self.block_size, audio_len)
            current_block_size = block_end - block_start
            
            # Get current audio block
            audio_block = audio[:, block_start:block_end]
            
            # Initialize block output
            block_output = torch.zeros_like(audio_block)
            
            # Apply each resonator and sum outputs
            for i in range(self.n_formants):
                # Extract parameters for this formant
                freq_i = formant_freqs[:, :, i]  # [B, C]
                bw_i = formant_bws[:, :, i]      # [B, C]
                amp_i = formant_amps[:, :, i]    # [B, C]
                
                # Apply resonator
                resonator_out, new_state = self.resonators[i](
                    audio_block, freq_i, bw_i, amp_i, states[i])
                
                # Update state for next block
                states[i] = new_state
                
                # Add to output (summing all formants)
                block_output += resonator_out
            
            # Copy block output to main output tensor
            output[:, block_start:block_end] = block_output
        
        return output


def apply_formant_filter_batch(audio, formant_params, sampling_rate, block_size=128):
    """
    Helper function to apply formant filtering in batch mode.
    
    Args:
        audio: Audio signals [B, T]
        formant_params: Tuple of (frequencies, bandwidths, amplitudes)
        sampling_rate: Sampling rate in Hz
        block_size: Size of processing blocks (smaller = less memory)
        
    Returns:
        Filtered audio [B, T]
    """
    formant_filter = FormantFilter(
        sampling_rate, 
        n_formants=formant_params[0].shape[-1],
        block_size=block_size
    )
    formant_freqs, formant_bws, formant_amps = formant_params
    return formant_filter(audio, formant_freqs, formant_bws, formant_amps)