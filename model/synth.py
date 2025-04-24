import torch
import torch.nn as nn
import math
import numpy as np

# Import the optimized modules
from ddsp.modules import HarmonicOscillator
from model.sing_vocoder import SingVocoder

from ddsp.core import scale_function, unit_to_hz2, frequency_filter, upsample, create_formant_filter, apply_vibrato

class Synth(nn.Module):
    """
    Enhanced synthesizer with formant modeling, vibrato, and voice quality parameters
    for improved vocal quality.
    """
    def __init__(self, 
            sampling_rate,
            block_size,
            n_mag_harmonic,
            n_mag_noise,
            n_harmonics,
            phone_map_len,
            singer_map_len,
            language_map_len,
            n_formants=4,
            n_breath_bands=8,
            n_mels=80,
            use_gradient_checkpointing=True):
        super().__init__()

        # Store parameters
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        self.n_mag_harmonic = n_mag_harmonic
        self.n_formants = n_formants
        
        # SingVocoder with enhanced formant modeling and singer adaptation
        self.sing_vocoder = SingVocoder(
            phone_map_len=phone_map_len,
            singer_map_len=singer_map_len,
            language_map_len=language_map_len,
            hidden_dim=256,
            n_harmonics=n_harmonics,
            n_mag_harmonic=n_mag_harmonic,
            n_mag_noise=n_mag_noise,
            n_formants=n_formants,
            use_checkpoint=use_gradient_checkpointing
        )
        
        self.harmonic_synthsizer = HarmonicOscillator(sampling_rate)

    def forward(self, batch, initial_phase=None):
        '''
        Predict control parameters from linguistic features and synthesize audio
        with enhanced vocal tract modeling, vibrato, and voice quality controls
        
        Args:
            batch: Dictionary containing:
                phone_seq_mel: B x n_frames x phone_dim
                f0: B x n_frames x 1
                singer_id: B x 1
                language_id: B x 1
                singer_weights: Optional B x n_singers (for singer mixing)
                singer_ids: Optional B x n_singers (for singer mixing)
            initial_phase: Optional initial phase for continuity
            
        Returns:
            signal: Synthesized audio
            f0: Predicted fundamental frequency
            final_phase: Final phase (for continuity in streaming)
            components: Tuple of (harmonic, noise, frame_rate_amplitudes)
        '''
        # Extract inputs from batch
        phone_seq = batch['phone_seq_mel']  # Use mel-aligned phonemes
        f0_in = batch['f0']
        singer_id = batch['singer_id']
        language_id = batch['language_id']
        
        # Support for singer mixing if provided
        singer_weights = batch.get('singer_weights', None)
        singer_ids = batch.get('singer_ids', None)
        
        # Get synthesis parameters from enhanced SingVocoder
        core_params = self.sing_vocoder(
            phone_seq, f0_in, singer_id, language_id,
            singer_weights=singer_weights, singer_ids=singer_ids
        )
        
        # Process f0
        f0_unit = core_params['f0']  # units
        f0_unit = torch.sigmoid(f0_unit)
        f0 = unit_to_hz2(f0_unit, hz_min=80.0, hz_max=1000.0)
        f0[f0<80] = 0

        # Store pitch for return
        pitch = f0.clone()
        
        # Get vibrato parameters
        vibrato_rate = torch.sigmoid(core_params['vibrato_rate']) * 8.0  # 0-8 Hz
        vibrato_depth = torch.sigmoid(core_params['vibrato_depth']) * 50.0  # 0-50 cents
        vibrato_delay = torch.sigmoid(core_params['vibrato_delay']) * 0.5  # 0-500ms delay
        
        # Get voice quality parameters
        breathiness = torch.sigmoid(core_params['breathiness'])
        tension = torch.sigmoid(core_params['tension'])
        
        # Process amplitude parameters
        A = scale_function(core_params['A'])
        amplitudes = scale_function(core_params['amplitudes'])
        src_param = scale_function(core_params['harmonic_magnitude'])
        noise_param = scale_function(core_params['noise_magnitude'])

        # Normalize amplitudes to distribution
        amplitudes /= amplitudes.sum(-1, keepdim=True) + 1e-8  # Add epsilon to prevent division by zero
        amplitudes *= A

        # Get formant parameters
        formant_freqs = core_params['formant_freqs']
        formant_widths = core_params['formant_widths']
        formant_gains = core_params['formant_gains']
        
        # Create formant filter
        formant_filter = create_formant_filter(
            formant_freqs, formant_widths, formant_gains,
            n_samples=self.n_mag_harmonic,
            sampling_rate=self.sampling_rate
        )

        # Get dimensions
        batch_size, n_frames, _ = pitch.shape
        
        # Save the original frame-rate amplitudes for loss calculation
        frame_rate_amplitudes = amplitudes.clone()

        # Upsample parameters to audio rate
        pitch = upsample(pitch, self.block_size)
        amplitudes = upsample(amplitudes, self.block_size)
        
        # Apply vibrato to the upsampled f0
        pitch_with_vibrato = apply_vibrato(
            pitch, 
            vibrato_rate, 
            vibrato_depth, 
            vibrato_delay,
            sampling_rate=self.sampling_rate
        )

        # Generate harmonic component
        harmonic, final_phase = self.harmonic_synthsizer(
            pitch_with_vibrato, amplitudes, initial_phase)
        
        # Apply source spectrum filtering
        harmonic = frequency_filter(
            harmonic,
            src_param
        )
        
        # Apply formant filtering for enhanced vocal tract modeling
        harmonic = frequency_filter(
            harmonic,
            formant_filter
        )
        
        # Apply tension effect (nonlinear shaping for vocal "pressure")
        if torch.any(tension > 0.05):
            tension_up = upsample(tension, self.block_size).squeeze(-1)
            # Simple waveshaping distortion controlled by tension parameter
            harmonic = harmonic * (1.0 + 0.3 * tension_up * torch.tanh(2.0 * harmonic))
            
        # Generate noise component with breathiness control
        noise = torch.rand_like(harmonic).to(noise_param) * 2 - 1
        noise = frequency_filter(
            noise,
            noise_param
        )
        
        # Apply breathiness control by scaling noise and mixing properly
        breathiness_up = upsample(breathiness, self.block_size).squeeze(-1)
        # Ensure breathiness doesn't completely eliminate noise in consonants
        scaled_noise = noise * (0.3 + 0.7 * breathiness_up)
        
        # Combine components
        signal = harmonic + scaled_noise

        return signal, f0, final_phase, (harmonic, scaled_noise, frame_rate_amplitudes)
    
    def set_gradient_checkpointing(self, enabled=True):
        """
        Enable or disable gradient checkpointing to trade compute for memory
        
        Args:
            enabled: Bool, whether to enable checkpointing
        """
        # Update the module's checkpointing flag
        if hasattr(self.sing_vocoder, 'use_checkpoint'):
            self.sing_vocoder.use_checkpoint = enabled
            
        return self