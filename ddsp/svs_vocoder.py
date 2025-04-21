# ddsp/svs_vocoder.py - Enhanced with formant filtering

import torch
import torch.nn as nn
import numpy as np

from ddsp.formant_filter import FormantFilter
from ddsp.modules import HarmonicOscillator
from ddsp.core import scale_function, upsample, frequency_filter
from ddsp.mel2control import Mel2Control

class SVSVocoder(nn.Module):
    def __init__(self, 
                 sampling_rate,
                 block_size,
                 n_mag_harmonic,
                 n_mag_noise,
                 n_harmonics,
                 num_phonemes,
                 num_singers,
                 num_languages,
                 n_mels=80,
                 n_formants=5):
        super().__init__()
        
        print(' [Model] Enhanced Singing Voice Synthesis (SVS) Vocoder with Formant Filtering')
        
        # Parameters
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        
        # Define control parameter structure
        self.control_splits = {
            'f0': 1, 
            'A': 1,
            'amplitudes': n_harmonics,
            'harmonic_magnitude': n_mag_harmonic,
            'noise_magnitude': n_mag_noise
        }
        
        # Import here to avoid circular imports
        from ddsp.pseudo_mel import PseudoMelGenerator, FormantParameterPredictor
        
        # Pseudo-mel generator
        self.pseudo_mel_generator = PseudoMelGenerator(
            num_phonemes=num_phonemes,
            num_singers=num_singers,
            num_languages=num_languages,
            hidden_dim=64,
            mel_dims=n_mels
        )
        
        # Formant parameter predictor
        self.formant_predictor = FormantParameterPredictor(
            num_phonemes=num_phonemes,
            hidden_dim=32,
            n_formants=n_formants,
            mel_dims=n_mels
        )
        
        # Mel2Control for synthesis parameters
        self.mel2ctrl = Mel2Control(n_mels, self.control_splits)
        
        # Harmonic Synthesizer
        self.harmonic_synthesizer = HarmonicOscillator(sampling_rate)
        
        # Formant Filter
        self.formant_filter = FormantFilter(sampling_rate)
        
    def forward(self, phonemes, durations, f0_in, singer_ids, language_ids, initial_phase=None):
        """
        Forward pass of the SVS vocoder with formant filtering
        
        Args:
            phonemes: Phoneme sequence [B, T_phones]
            durations: Phoneme durations [B, T_phones]
            f0_in: Fundamental frequency [B, T_frames]
            singer_ids: Singer IDs [B]
            language_ids: Language IDs [B]
            initial_phase: Initial phase for harmonic oscillator (optional)
            
        Returns:
            signal: Synthesized audio
            f0: F0 used for synthesis
            phase: Final phase state
            components: Tuple of (harmonic, noise) components
            formant_params: Tuple of formant parameters
        """
        # Generate pseudo-mel representation
        pseudo_mel, hidden_states = self.pseudo_mel_generator(
            phonemes, durations, f0_in, singer_ids, language_ids)
        
        # Predict formant parameters
        formant_params = self.formant_predictor(pseudo_mel, hidden_states, phonemes)
        
        # Generate control parameters from pseudo-mel
        ctrls = self.mel2ctrl(pseudo_mel)
        
        # Extract and process control parameters
        f0 = ctrls['f0']
        A = scale_function(ctrls['A'])
        amplitudes = scale_function(ctrls['amplitudes'])
        harmonic_magnitude = scale_function(ctrls['harmonic_magnitude'])
        noise_magnitude = scale_function(ctrls['noise_magnitude'])
        
        # Normalize amplitudes
        amplitudes = amplitudes / (amplitudes.sum(-1, keepdim=True) + 1e-8)
        amplitudes = amplitudes * A
        
        # Upsample parameters
        f0 = upsample(f0, self.block_size)
        amplitudes = upsample(amplitudes, self.block_size)
        
        # Generate harmonic component
        harmonic, phase = self.harmonic_synthesizer(f0, amplitudes, initial_phase)
        
        # Apply harmonic shaping
        harmonic = frequency_filter(harmonic, harmonic_magnitude)
        
        # Apply formant filtering to harmonic component
        harmonic = self.formant_filter(harmonic, *formant_params)
        
        # Generate and shape noise component
        noise = torch.randn_like(harmonic) * 2 - 1
        noise = frequency_filter(noise, noise_magnitude)
        
        # Combine signals
        signal = harmonic + noise
        
        return signal, f0, phase, (harmonic, noise), formant_params
        
    def inference(self, phonemes, durations, f0, singer_id, language_id, initial_phase=None):
        """
        Simplified interface for inference
        
        Args:
            phonemes: Phoneme sequence [B, T_phones]
            durations: Phoneme durations [B, T_phones]
            f0: Fundamental frequency [B, T_frames]
            singer_id: Singer ID [B]
            language_id: Language ID [B]
            initial_phase: Initial phase for harmonic oscillator (optional)
            
        Returns:
            audio: Synthesized audio
        """
        signal, _, _, _, _ = self.forward(
            phonemes, durations, f0, singer_id, language_id, initial_phase)
        return signal