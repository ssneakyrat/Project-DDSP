import torch
import torch.nn as nn

from ddsp.melception import EnhancedMelception
from ddsp.mel2control import Mel2Control
from ddsp.modules import HarmonicOscillator
from ddsp.vocal_oscillator import VocalOscillator
from ddsp.core import scale_function, unit_to_hz2, frequency_filter, upsample

class Synth(nn.Module):
    def __init__(self, 
            sampling_rate,
            block_size,
            n_mag_harmonic,
            n_mag_noise,
            n_harmonics,
            n_mels=80,
            n_formants=4,
            n_breath_bands=8):
        super().__init__()

        # params
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        
        # Define enhanced output splits for Melception
        split_map = {
            # Basic parameters (original)
            'f0': 1, 
            'A': 1,
            'amplitudes': n_harmonics,
            'harmonic_magnitude': n_mag_harmonic,
            'noise_magnitude': n_mag_noise,
            
            # Vibrato parameters
            'vibrato_rate': 1,
            'vibrato_depth': 1,
            
            # Formant parameters
            'formant_freqs': n_formants,
            'formant_bandwidths': n_formants,
            'formant_gains': n_formants,
            
            # Voice quality parameters
            'glottal_open_quotient': 1,
            'phase_coherence': 1,
            'breathiness': 1,
            'breath_spectral_shape': n_breath_bands
        }
        
        # Use enhanced Melception implementation
        self.mel2ctrl = EnhancedMelception(n_mels, split_map)

        # Vocal Oscillator (replaces the basic Harmonic Synthesizer)
        self.vocal_synthsizer = VocalOscillator(sampling_rate, n_formants=n_formants)
        
        # Store configuration
        self.n_formants = n_formants
        self.n_breath_bands = n_breath_bands

    def process_controls(self, ctrls):
        """Apply appropriate scaling and constraints to each control parameter"""
        
        # Basic parameters (existing)
        f0_unit = ctrls['f0']
        f0_unit = torch.sigmoid(f0_unit)
        f0 = unit_to_hz2(f0_unit, hz_min=80.0, hz_max=1000.0)
        f0[f0<80] = 0
        
        A = scale_function(ctrls['A'])
        amplitudes = scale_function(ctrls['amplitudes'])
        harmonic_magnitude = scale_function(ctrls['harmonic_magnitude'])
        noise_magnitude = scale_function(ctrls['noise_magnitude'])
        
        # Normalize amplitudes to distribution
        amplitudes = amplitudes / (amplitudes.sum(-1, keepdim=True) + 1e-8)
        amplitudes = amplitudes * A
        
        # Vibrato (constrain to reasonable ranges)
        vibrato_rate = torch.sigmoid(ctrls['vibrato_rate']) * 8.0 + 3.0  # 3-11 Hz
        vibrato_depth = torch.sigmoid(ctrls['vibrato_depth']) * 0.5  # 0-0.5 semitones
        
        # Formants (ensure proper ordering and ranges)
        # Start with base formant positions and adjust
        device = f0.device
        batch_size = f0.shape[0]
        base_formants = torch.tensor([500.0, 1500.0, 2500.0, 3500.0]).to(device)
        formant_adjustment = torch.tanh(ctrls['formant_freqs']) * 500.0  # ±500 Hz adjustment
        formant_freqs = base_formants.reshape(1, 1, -1) + formant_adjustment
        
        # Ensure formant frequencies are properly ordered (increasing)
        # Using cumulative max to ensure ascending order
        formant_freqs_flat = formant_freqs.reshape(batch_size, -1)
        formant_freqs_sorted = torch.cummax(formant_freqs_flat, dim=1)[0]
        formant_freqs = formant_freqs_sorted.reshape(formant_freqs.shape)
        
        # Ensure minimum bandwidths and proper scaling
        formant_bandwidths = torch.sigmoid(ctrls['formant_bandwidths']) * 200.0 + 50.0  # 50-250 Hz
        
        # Scale gains and normalize
        formant_gains = torch.softmax(ctrls['formant_gains'], dim=-1) * 4.0  # Normalized distribution
        
        # Voice quality parameters
        glottal_open_quotient = torch.sigmoid(ctrls['glottal_open_quotient']) * 0.4 + 0.4  # 0.4-0.8
        phase_coherence = torch.sigmoid(ctrls['phase_coherence'])  # 0-1
        breathiness = torch.sigmoid(ctrls['breathiness']) * 0.5  # 0-0.5
        
        # Breath spectral shape (ensure decreasing energy at higher frequencies)
        breath_spectral_shape = torch.softmax(-torch.abs(ctrls['breath_spectral_shape']), dim=-1)
        
        # Return processed controls
        return {
            'f0': f0,
            'amplitudes': amplitudes,
            'harmonic_magnitude': harmonic_magnitude,
            'noise_magnitude': noise_magnitude,
            'vibrato_rate': vibrato_rate,
            'vibrato_depth': vibrato_depth,
            'formant_freqs': formant_freqs,
            'formant_bandwidths': formant_bandwidths,
            'formant_gains': formant_gains,
            'glottal_open_quotient': glottal_open_quotient,
            'phase_coherence': phase_coherence,
            'breathiness': breathiness,
            'breath_spectral_shape': breath_spectral_shape
        }

    def forward(self, mel, initial_phase=None):
        '''
            mel: B x n_frames x n_mels
        '''
        # Extract control parameters from mel spectrogram
        raw_ctrls = self.mel2ctrl(mel)
        
        # Process and constrain control parameters
        ctrls = self.process_controls(raw_ctrls)

        # Extract parameters
        f0 = ctrls['f0']
        amplitudes = ctrls['amplitudes']
        harmonic_magnitude = ctrls['harmonic_magnitude']
        noise_magnitude = ctrls['noise_magnitude']
        
        # Extract advanced vocal parameters
        vibrato_rate = ctrls['vibrato_rate'] 
        vibrato_depth = ctrls['vibrato_depth']
        formant_freqs = ctrls['formant_freqs']
        formant_bandwidths = ctrls['formant_bandwidths']
        formant_gains = ctrls['formant_gains']
        glottal_open_quotient = ctrls['glottal_open_quotient']
        phase_coherence = ctrls['phase_coherence']
        breathiness = ctrls['breathiness']
        breath_spectral_shape = ctrls['breath_spectral_shape']

        # Save pitch for output
        pitch = f0
        
        # Prepare shapes
        B, n_frames, _ = pitch.shape
        
        # Upsample time-varying parameters to audio rate
        pitch = upsample(pitch, self.block_size)
        amplitudes = upsample(amplitudes, self.block_size)
        
        # Upsample other time-varying control parameters
        vibrato_rate = upsample(vibrato_rate, self.block_size)
        vibrato_depth = upsample(vibrato_depth, self.block_size)
        glottal_open_quotient = upsample(glottal_open_quotient, self.block_size)
        phase_coherence = upsample(phase_coherence, self.block_size)
        breathiness = upsample(breathiness, self.block_size)
        
        # Upsample formant parameters
        formant_freqs = upsample(formant_freqs, self.block_size)
        formant_bandwidths = upsample(formant_bandwidths, self.block_size)
        formant_gains = upsample(formant_gains, self.block_size)

        # Generate vocal synthesis using the enhanced VocalOscillator
        harmonic, final_phase = self.vocal_synthsizer(
            f0=pitch, 
            amplitudes=amplitudes, 
            vibrato_rate=vibrato_rate,
            vibrato_depth=vibrato_depth,
            formant_freqs=formant_freqs,
            formant_bandwidths=formant_bandwidths,
            formant_gains=formant_gains,
            glottal_open_quotient=glottal_open_quotient,
            phase_coherence=phase_coherence,
            breathiness=breathiness,
            breath_spectral_shape=breath_spectral_shape,
            initial_phase=initial_phase
        )
        
        # Apply frequency filter for harmonic component
        harmonic = frequency_filter(harmonic, harmonic_magnitude)
            
        # Generate additional noise component for extra flexibility
        noise = torch.rand_like(harmonic).to(noise_magnitude) * 2 - 1
        noise = frequency_filter(noise, noise_magnitude)
        
        # Combine harmonic and noise components
        signal = harmonic + noise

        return signal, f0, final_phase, (harmonic, noise)