import torch
import torch.nn as nn

# Import the optimized modules
from ddsp.melception import Melception
from ddsp.expression_predictor import ExpressionPredictor
#from ddsp.vocal_oscillator import VocalOscillator
from ddsp.standalone_vocoder import AbstractVocalSynthesizer, VocalSynthConfig, PrecisionMode, FormantQuality, BreathQuality

from ddsp.core import scale_function, unit_to_hz2, frequency_filter, upsample

class Synth(nn.Module):
    """
    Enhanced synthesizer with separate parameter predictors for improved vocal quality
    and reduced memory footprint.
    """
    def __init__(self, 
            sampling_rate,
            block_size,
            n_mag_harmonic,
            n_mag_noise,
            n_harmonics,
            n_formants=4,
            n_breath_bands=8,
            n_mels=80,
            use_gradient_checkpointing=True):
        super().__init__()

        # Store parameters
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        
        # Core parameter predictor (optimized for memory efficiency)
        core_split_map = {
            'f0': 1, 
            'A': 1,
            'amplitudes': n_harmonics,
            'harmonic_magnitude': n_mag_harmonic,
            'noise_magnitude': n_mag_noise
        }
        self.mel2ctrl = Melception(
            input_channel=n_mels, 
            output_splits=core_split_map,
            dim_model=48,  # Reduced from default 64
            use_checkpoint=use_gradient_checkpointing
        )

        # Dedicated expression parameter predictor for vocal characteristics
        self.expression_predictor = ExpressionPredictor(
            input_dim=n_mels,
            hidden_dim=48,
            n_formants=n_formants,
            n_breath_bands=n_breath_bands,
            use_checkpoint=use_gradient_checkpointing
        )

        # Advanced vocal synthesizer
        #self.vocal_synthsizer = VocalOscillator(sampling_rate)
        # Advanced vocal synthesizer (using the new standalone implementation)
        vocoder_config = VocalSynthConfig(
            sampling_rate=sampling_rate,
            n_formants=n_formants,
            precision=PrecisionMode.FLOAT32,
            formant_quality=FormantQuality.NORMAL,
            breath_quality=BreathQuality.NORMAL,
            use_gpu=True,
            fallback_to_cpu=False  # Allow fallback to CPU if GPU not available
        )
        
        self.vocal_synthesizer = AbstractVocalSynthesizer.create(config=vocoder_config)

    def forward(self, mel, initial_phase=None):
        '''
        Predict control parameters from mel spectrogram and synthesize audio
        
        Args:
            mel: B x n_frames x n_mels
            initial_phase: Optional initial phase for continuity
            
        Returns:
            signal: Synthesized audio
            f0: Predicted fundamental frequency
            final_phase: Final phase (for continuity in streaming)
            components: Tuple of (harmonic, noise) components
        '''
        # Get core synthesis parameters
        core_params = self.mel2ctrl(mel)

        # Get expression parameters
        expression_params = self.expression_predictor(mel)
        
        # Unpack and process core parameters
        f0_unit = core_params['f0']  # units
        f0_unit = torch.sigmoid(f0_unit)
        f0 = unit_to_hz2(f0_unit, hz_min=80.0, hz_max=1000.0)
        f0[f0<80] = 0

        # Store pitch for return
        pitch = f0
        
        # Process amplitude parameters
        A = scale_function(core_params['A'])
        amplitudes = scale_function(core_params['amplitudes'])
        src_param = scale_function(core_params['harmonic_magnitude'])
        noise_param = scale_function(core_params['noise_magnitude'])

        # Normalize amplitudes to distribution
        amplitudes /= amplitudes.sum(-1, keepdim=True) + 1e-8  # Add epsilon to prevent division by zero
        amplitudes *= A

        # Get dimensions
        B, n_frames, _ = pitch.shape
        
        # Upsample to audio rate
        pitch = upsample(pitch, self.block_size)
        amplitudes = upsample(amplitudes, self.block_size)
        
        # Upsample expression parameters
        expression_upsampled = {}
        for key, value in expression_params.items():
            expression_upsampled[key] = upsample(value, self.block_size)
        '''
        # Synthesize harmonic component with advanced vocal parameters
        harmonic, final_phase = self.vocal_synthsizer(
            pitch, 
            amplitudes, 
            vibrato_rate=expression_upsampled['vibrato_rate'],
            vibrato_depth=expression_upsampled['vibrato_depth'],
            formant_freqs=expression_upsampled['formant_freqs'],
            formant_bandwidths=expression_upsampled['formant_bandwidths'],
            formant_gains=expression_upsampled['formant_gains'],
            glottal_open_quotient=expression_upsampled['glottal_open_quotient'],
            phase_coherence=expression_upsampled['phase_coherence'],
            breathiness=expression_upsampled['breathiness'],
            breath_spectral_shape=expression_upsampled['breath_spectral_shape'],
            initial_phase=initial_phase
        )
        '''
        # Synthesize harmonic component with advanced vocal parameters
        harmonic, final_phase = self.vocal_synthesizer(
            pitch, 
            amplitudes, 
            initial_phase=initial_phase,
            vibrato_rate=expression_upsampled['vibrato_rate'],
            vibrato_depth=expression_upsampled['vibrato_depth'],
            formant_freqs=expression_upsampled['formant_freqs'],
            formant_bandwidths=expression_upsampled['formant_bandwidths'],
            formant_gains=expression_upsampled['formant_gains'],
            glottal_open_quotient=expression_upsampled['glottal_open_quotient'],
            phase_coherence=expression_upsampled['phase_coherence'],
            breathiness=expression_upsampled['breathiness'],
            breath_spectral_shape=expression_upsampled['breath_spectral_shape']
        )
        # Apply spectral filtering to harmonic component
        harmonic = frequency_filter(
            harmonic,
            src_param
        )
            
        # Generate noise component (breath noise is already handled in VocalOscillator)
        noise = torch.rand_like(harmonic).to(noise_param) * 2 - 1
        noise = frequency_filter(
            noise,
            noise_param
        )
        
        # Combine components
        signal = harmonic + noise

        return signal, f0, final_phase, (harmonic, noise)
    
    def set_gradient_checkpointing(self, enabled=True):
        """
        Enable or disable gradient checkpointing to trade compute for memory
        
        Args:
            enabled: Bool, whether to enable checkpointing
        """
        # Update the modules' checkpointing flags
        if hasattr(self.mel2ctrl, 'use_checkpoint'):
            self.mel2ctrl.use_checkpoint = enabled
            
        if hasattr(self.expression_predictor, 'use_checkpoint'):
            self.expression_predictor.use_checkpoint = enabled
            
        return self