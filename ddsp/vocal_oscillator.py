import torch
import torch.nn as nn
import numpy as np
from ddsp.core import remove_above_nyquist

class VocalOscillator(nn.Module):
    """
    Enhanced oscillator for realistic vocal synthesis with vibrato, formants, glottal flow, and breathiness
    
    This extends the original HarmonicOscillator with vocal-specific features:
    - Vibrato modeling
    - Formant filtering
    - Glottal flow model
    - Phase coherence control
    - Breathiness control
    """
    def __init__(
            self, 
            fs, 
            n_formants=4,
            is_remove_above_nyquist=True,
            default_vibrato_rate=5.5,
            default_vibrato_depth=0.1,
            default_breathiness=0.1):
        super().__init__()
        self.fs = fs
        self.is_remove_above_nyquist = is_remove_above_nyquist
        
        # Default parameters for vibrato
        self.default_vibrato_rate = default_vibrato_rate   # Hz
        self.default_vibrato_depth = default_vibrato_depth # semitones
        
        # Default breathiness level
        self.default_breathiness = default_breathiness
        
        # Default formant values (soprano)
        self.default_formant_freqs = torch.tensor([800.0, 1150.0, 2900.0, 3900.0])
        self.default_formant_bandwidths = torch.tensor([80.0, 90.0, 120.0, 130.0])
        self.default_formant_gains = torch.tensor([1.0, 0.5, 0.25, 0.1])
        
        # Initialize glottal flow model parameters
        self.default_glottal_open_quotient = 0.7  # Open phase duration relative to cycle
        self.default_glottal_spectral_tilt = -12  # dB/octave

    def _apply_vibrato(self, f0, vibrato_rate, vibrato_depth):
        """
        Apply vibrato to the fundamental frequency
        
        Args:
            f0: B x T x 1 tensor with fundamental frequencies in Hz
            vibrato_rate: B x 1 x 1 tensor with vibrato rates in Hz
            vibrato_depth: B x 1 x 1 tensor with vibrato depth in semitones
            
        Returns:
            f0_vibrato: B x T x 1 tensor with modulated frequencies
        """
        batch_size, time_steps, _ = f0.shape
        
        # Create time vector
        time = torch.arange(0, time_steps).to(f0.device) / self.fs
        time = time.view(1, -1, 1).repeat(batch_size, 1, 1)
        
        # Apply sinusoidal vibrato
        # Convert semitones to multiplication factor: 2^(depth/12)
        vibrato_factor = 2.0 ** (vibrato_depth / 12.0)
        
        # Modulating between 1/factor and factor using sine wave
        vibrato_multiplier = torch.exp(
            torch.sin(2 * torch.pi * vibrato_rate * time) * 
            torch.log(vibrato_factor)
        )
        
        # Apply to f0, only where f0 > 0 (voiced regions)
        f0_vibrato = f0 * vibrato_multiplier
        f0_vibrato = torch.where(f0 > 0, f0_vibrato, f0)
        
        return f0_vibrato

    def _glottal_flow_model(self, phase, open_quotient):
        """
        Model the glottal flow waveform based on phase
        
        Args:
            phase: B x T x n_harmonic tensor with phases for each harmonic
            open_quotient: B x 1 x 1 tensor with open quotient (0.0-1.0)
            
        Returns:
            flow: B x T x n_harmonic tensor with glottal flow waveforms
        """
        # Normalize phase to [0, 1] range
        norm_phase = (phase % (2 * torch.pi)) / (2 * torch.pi)
        
        # LF-model inspired glottal flow (simplified)
        # Rising phase (open phase)
        rising = norm_phase / open_quotient
        rising_flow = torch.sin(rising * torch.pi / 2.0) ** 2
        rising_mask = (norm_phase < open_quotient).float()
        
        # Falling phase (return phase)
        falling = (norm_phase - open_quotient) / (1.0 - open_quotient)
        falling_flow = torch.cos(falling * torch.pi / 2.0) ** 2
        falling_mask = (norm_phase >= open_quotient).float()
        
        # Combine phases
        flow = rising_flow * rising_mask + falling_flow * falling_mask
        
        # Adjust DC offset to ensure mean is 0
        flow = flow - 0.5
        
        return flow
    
    def _apply_phase_coherence(self, phases, coherence_factor):
        """
        Adjust phase relationships between harmonics
        
        Args:
            phases: B x T x n_harmonic tensor with phases
            coherence_factor: B x 1 x 1 tensor with coherence factor (0.0-1.0)
            
        Returns:
            modified_phases: B x T x n_harmonic tensor with adjusted phases
        """
        # Extract fundamental phase
        fundamental_phase = phases[:, :, 0:1]
        
        # For higher harmonics, blend between independent phase and 
        # phase that would be coherent with fundamental
        if phases.size(2) > 1:
            harmonic_indices = torch.arange(1, phases.size(2)).to(phases.device)
            harmonic_indices = harmonic_indices.view(1, 1, -1)
            
            coherent_phases = fundamental_phase * harmonic_indices
            
            # Apply coherence blending to harmonics 2+
            higher_harmonics = phases[:, :, 1:]
            blended_phases = (
                coherence_factor * coherent_phases + 
                (1 - coherence_factor) * higher_harmonics
            )
            
            # Recombine with fundamental
            modified_phases = torch.cat([fundamental_phase, blended_phases], dim=2)
        else:
            modified_phases = phases
            
        return modified_phases
    
    def _apply_formant_filter(self, signal, formant_freqs, formant_bandwidths, formant_gains):
        """
        Apply formant filtering using frequency domain processing
        
        Args:
            signal: B x T tensor with input signal
            formant_freqs: B x n_formants tensor with formant frequencies
            formant_bandwidths: B x n_formants tensor with formant bandwidths
            formant_gains: B x n_formants tensor with formant gains
            
        Returns:
            filtered_signal: B x T tensor with formant-filtered signal
        """
        batch_size, time_steps = signal.shape
        
        # Get frequency domain representation
        signal_fft = torch.fft.rfft(signal)
        freqs = torch.fft.rfftfreq(time_steps, d=1.0/self.fs).to(signal.device)
        
        # Initialize frequency response
        freq_response = torch.zeros(batch_size, len(freqs)).to(signal.device)
        
        # Calculate frequency response for each formant
        for i in range(formant_freqs.size(1)):
            for b in range(batch_size):
                # Parameters for this formant
                f0 = formant_freqs[b, i]
                bw = formant_bandwidths[b, i]
                gain = formant_gains[b, i]
                
                # Create resonance with second-order bandpass response
                response = gain * torch.pow(
                    1.0 / (1.0 + torch.pow((freqs - f0) / (bw/2), 2)), 
                    1.0
                )
                
                # Add to total response
                freq_response[b] += response
        
        # Apply filter
        filtered_fft = signal_fft * freq_response
        
        # Convert back to time domain
        filtered_signal = torch.fft.irfft(filtered_fft, n=time_steps)
        
        return filtered_signal
    
    def _generate_breath_noise(self, shape, spectral_shape):
        """
        Generate breath noise with specified spectral shape
        
        Args:
            shape: tuple with shape of noise to generate (B, T)
            spectral_shape: B x n_bands tensor controlling spectral shape
            
        Returns:
            breath_noise: B x T tensor with shaped noise
        """
        batch_size, time_steps = shape
        
        # Generate white noise
        noise = torch.randn(shape).to(spectral_shape.device)
        
        # Shape spectrum using simple filter bank
        n_bands = spectral_shape.size(1)
        bands_per_octave = 3
        min_freq = 80.0  # Hz
        
        # Create filter bank
        filtered_noise = torch.zeros_like(noise)
        
        for b in range(batch_size):
            # FFT of noise
            noise_fft = torch.fft.rfft(noise[b])
            freqs = torch.fft.rfftfreq(time_steps, d=1.0/self.fs).to(noise.device)
            
            # Apply spectral shape
            filter_response = torch.zeros(len(freqs)).to(noise.device)
            
            for i in range(n_bands):
                center_freq = min_freq * (2.0 ** (i / bands_per_octave))
                bandwidth = center_freq * (2.0 ** (0.5 / bands_per_octave)) - center_freq
                
                # Apply Gaussian shape for each band
                freq_response = torch.exp(
                    -0.5 * ((freqs - center_freq) / bandwidth) ** 2
                ) * spectral_shape[b, i]
                
                filter_response += freq_response
                
            # Apply filter
            filtered_fft = noise_fft * filter_response
            
            # IFFT back to time domain
            filtered_noise[b] = torch.fft.irfft(filtered_fft, n=time_steps)
            
            # Normalize
            filtered_noise[b] = filtered_noise[b] / (torch.std(filtered_noise[b]) + 1e-8)
            
        return filtered_noise
        
    def forward(self, f0, amplitudes, 
                vibrato_rate=None, vibrato_depth=None,
                formant_freqs=None, formant_bandwidths=None, formant_gains=None,
                glottal_open_quotient=None, phase_coherence=None, 
                breathiness=None, breath_spectral_shape=None,
                initial_phase=None):
        '''
        Args:
            f0: B x T x 1 (Hz)
            amplitudes: B x T x n_harmonic
            vibrato_rate: B x 1 x 1 (Hz) or None
            vibrato_depth: B x 1 x 1 (semitones) or None
            formant_freqs: B x n_formants or None
            formant_bandwidths: B x n_formants or None
            formant_gains: B x n_formants or None
            glottal_open_quotient: B x 1 x 1 or None
            phase_coherence: B x 1 x 1 or None
            breathiness: B x 1 x 1 or None
            breath_spectral_shape: B x n_bands or None
            initial_phase: B x 1 x 1 or None
          
        Returns:
            signal: B x T
            final_phase: B x 1 x 1
        '''
        batch_size = f0.shape[0]
        device = f0.device
        
        # Initialize default parameters if not provided
        if initial_phase is None:
            initial_phase = torch.zeros(batch_size, 1, 1).to(device)
            
        # Vibrato parameters
        if vibrato_rate is None:
            vibrato_rate = torch.ones(batch_size, 1, 1).to(device) * self.default_vibrato_rate
        if vibrato_depth is None:
            vibrato_depth = torch.ones(batch_size, 1, 1).to(device) * self.default_vibrato_depth
            
        # Formant parameters
        n_formants = self.default_formant_freqs.size(0)
        if formant_freqs is None:
            formant_freqs = self.default_formant_freqs.view(1, n_formants).repeat(batch_size, 1).to(device)
        if formant_bandwidths is None:
            formant_bandwidths = self.default_formant_bandwidths.view(1, n_formants).repeat(batch_size, 1).to(device)
        if formant_gains is None:
            formant_gains = self.default_formant_gains.view(1, n_formants).repeat(batch_size, 1).to(device)
            
        # Glottal model parameters
        if glottal_open_quotient is None:
            glottal_open_quotient = torch.ones(batch_size, 1, 1).to(device) * self.default_glottal_open_quotient
        if phase_coherence is None:
            phase_coherence = torch.ones(batch_size, 1, 1).to(device) * 0.5  # Default 50% coherence
            
        # Breathiness parameters
        if breathiness is None:
            breathiness = torch.ones(batch_size, 1, 1).to(device) * self.default_breathiness
        if breath_spectral_shape is None:
            # Default: decreasing energy at higher frequencies
            n_bands = 8
            breath_spectral_shape = torch.pow(
                0.7, 
                torch.arange(n_bands).view(1, -1).repeat(batch_size, 1)
            ).to(device)
        
        # Get voiced mask
        mask = (f0 > 0).detach()
        
        # 1. Apply vibrato to f0
        f0_vibrato = self._apply_vibrato(f0.detach(), vibrato_rate, vibrato_depth)
        
        # 2. Calculate phase
        phase = torch.cumsum(2 * torch.pi * f0_vibrato / self.fs, axis=1) + initial_phase
        n_harmonic = amplitudes.shape[-1]
        
        # Generate harmonic phases
        harmonic_indices = torch.arange(1, n_harmonic + 1).to(device)
        phases = phase * harmonic_indices.view(1, 1, -1)
        
        # 3. Apply phase coherence
        phases = self._apply_phase_coherence(phases, phase_coherence)
        
        # 4. Apply anti-aliasing if needed
        if self.is_remove_above_nyquist:
            amp = remove_above_nyquist(amplitudes, f0_vibrato, self.fs)
        else:
            amp = amplitudes.to(device)
        
        # 5. Use glottal flow instead of sinusoid
        excitation = self._glottal_flow_model(phases, glottal_open_quotient)
        harmonic = (excitation * amp).sum(-1)
        
        # 6. Apply formant filtering
        harmonic = self._apply_formant_filter(
            harmonic, formant_freqs, formant_bandwidths, formant_gains)
        
        # 7. Generate and add breath noise
        breath_noise = self._generate_breath_noise(
            (batch_size, harmonic.size(1)), 
            breath_spectral_shape
        )
        
        # Scale noise by breathiness
        scaled_noise = breath_noise * breathiness.view(batch_size, 1)
        
        # 8. Mix harmonic and noise components
        # Apply voiced mask to harmonic
        harmonic = harmonic * mask.squeeze(-1)
        
        # Final signal (mix harmonic and breath components)
        signal = harmonic + scaled_noise
        
        # Return final phase for continuity
        final_phase = phase[:, -1:, :] % (2 * torch.pi)
        
        return signal, final_phase.detach()
