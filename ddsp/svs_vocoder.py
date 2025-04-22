# ddsp/svs_vocoder.py - Enhanced with formant filtering and main.py compatibility

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
            mel_dims=n_mels
        )
        
        # Formant parameter predictor
        self.formant_predictor = FormantParameterPredictor(
            input_channel=n_mels,
            hidden_dim=32,
            n_formants=n_formants,
        )
        
        # Mel2Control for synthesis parameters (for standard mel input)
        self.mel2ctrl = Mel2Control(n_mels, self.control_splits)
        
        # Harmonic Synthesizer
        self.harmonic_synthesizer = HarmonicOscillator(sampling_rate)
        
        # Formant Filter
        self.formant_filter = FormantFilter(sampling_rate)
        
    def forward(self, using_svs_model, phonemes=None, f0_in=None, singer_ids=None, language_ids=None, initial_phase=None, mel=None):
        """
        Forward pass of the SVS vocoder with formant filtering
        
        This method is overloaded to support two modes:
        1. Full SVS mode with linguistic inputs (phonemes, durations, etc.)
        2. Compatibility mode with just mel input (for main.py)
        
        Args:
            phonemes: Phoneme sequence [B, T_phones] (optional)
            durations: Phoneme durations [B, T_phones] (optional)
            f0_in: Fundamental frequency [B, T_frames, 1] (optional)
            singer_ids: Singer IDs [B] (optional)
            language_ids: Language IDs [B] (optional)
            initial_phase: Initial phase for harmonic oscillator (optional)
            mel: Mel spectrogram [B, T, n_mels] (optional)
            
        Returns:
            signal: Synthesized audio
            f0: F0 used for synthesis
            phase: Final phase state
            components: Tuple of (harmonic, noise) components
            formant_params: Tuple of formant parameters
        """
        # Handling different input modes
        if using_svs_model:
            # Full SVS mode with linguistic inputs
            return self.forward_svs(phonemes, f0_in, singer_ids, language_ids, initial_phase)
        else:
            # Compatibility mode: main.py with just mel input
            return self.forward_mel(mel, initial_phase)
            
    def forward_svs(self, phonemes, f0_in, singer_ids, language_ids, initial_phase=None):
        """
        Full SVS forward pass with linguistic inputs
        """
        # Generate pseudo-mel representation
        pseudo_mel = self.pseudo_mel_generator(
            phonemes, f0_in, singer_ids, language_ids)
        
        # Predict formant parameters
        formant_params = self.formant_predictor(pseudo_mel)
        
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
    
    def forward_mel(self, mel, initial_phase=None):
        """
        Compatibility mode forward pass with just mel input (for main.py)
        """
        # Use mel2ctrl directly for standard ddsp parameters
        ctrls = self.mel2ctrl(mel)
        
        # Process control parameters as in standard ddsp models
        f0_unit = ctrls['f0']
        f0_unit = torch.sigmoid(f0_unit)
        f0 = torch.where(f0_unit > 0.05, 
                        f0_unit * 1000.0,  # Scale to 0-1000 Hz range
                        torch.zeros_like(f0_unit))
        
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
        
        # Create simple formant parameters (neutral formants)
        batch_size, seq_len = f0.shape[0], f0.shape[1]
        device = f0.device
        
        # Default formant frequencies for neutral vowel (in Hz)
        f1, f2, f3, f4, f5 = 500, 1500, 2500, 3500, 4500
        formant_freqs = torch.tensor([f1, f2, f3, f4, f5], device=device)
        formant_freqs = formant_freqs.expand(batch_size, seq_len, 5)
        
        # Default bandwidths (in Hz)
        bw1, bw2, bw3, bw4, bw5 = 80, 100, 120, 150, 200
        formant_bws = torch.tensor([bw1, bw2, bw3, bw4, bw5], device=device)
        formant_bws = formant_bws.expand(batch_size, seq_len, 5)
        
        # Equal amplitudes for all formants
        formant_amps = torch.ones(batch_size, seq_len, 5, device=device) / 5.0
        
        formant_params = (formant_freqs, formant_bws, formant_amps)
        
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
        signal, _, _, _, _ = self.forward_svs(
            phonemes, durations, f0, singer_id, language_id, initial_phase)
        return signal