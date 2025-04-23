import torch
import torch.nn as nn

# Import the optimized modules
from ddsp.modules import HarmonicOscillator
from model.sing_vocoder import SingVocoder
#from ddsp.standalone_vocoder import AbstractVocalSynthesizer, VocalSynthConfig, PrecisionMode, FormantQuality, BreathQuality

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
        
        # Replace Melception and ExpressionPredictor with SingVocoder
        self.sing_vocoder = SingVocoder(
            phone_map_len=phone_map_len,
            singer_map_len=singer_map_len,
            language_map_len=language_map_len,
            hidden_dim=256,
            n_harmonics=n_harmonics,
            n_mag_harmonic=n_mag_harmonic,
            n_mag_noise=n_mag_noise,
            use_checkpoint=use_gradient_checkpointing
        )

        # Advanced vocal synthesizer
        #self.vocal_synthsizer = VocalOscillator(sampling_rate)
        # Advanced vocal synthesizer (using the new standalone implementation)
        '''
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
        '''
        self.harmonic_synthsizer = HarmonicOscillator(sampling_rate)


    def forward(self, batch, initial_phase=None):
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
        # Extract inputs from batch
        phone_seq = batch['phone_seq_mel']  # Use mel-aligned phonemes
        f0_in = batch['f0']
        singer_id = batch['singer_id']
        language_id = batch['language_id']
        
        # Get synthesis parameters from SingVocoder
        core_params = self.sing_vocoder(phone_seq, f0_in, singer_id, language_id)
        
        # Process f0
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
        batch_size, n_frames, _ = pitch.shape
        
        # Upsample to audio rate
        pitch = upsample(pitch, self.block_size)
        amplitudes = upsample(amplitudes, self.block_size)

        # harmonic
        harmonic, final_phase = self.harmonic_synthsizer(
            pitch, amplitudes, initial_phase)
        harmonic = frequency_filter(
                        harmonic,
                        src_param)
        '''
        # Synthesize harmonic component with vocal parameters
        harmonic, final_phase = self.vocal_synthesizer(
            pitch, 
            amplitudes, 
            initial_phase=initial_phase
            #vibrato_rate=expression_params['vibrato_rate'],
            #vibrato_depth=expression_params['vibrato_depth'],
            #formant_freqs=expression_params['formant_freqs'],
            #formant_bandwidths=expression_params['formant_bandwidths'],
            #formant_gains=expression_params['formant_gains'],
            #glottal_open_quotient=expression_params['glottal_open_quotient'],
            #phase_coherence=expression_params['phase_coherence'],
            #breathiness=expression_params['breathiness'],
            #breath_spectral_shape=expression_params['breath_spectral_shape']
        )
        '''
        # Apply spectral filtering to harmonic component
        harmonic = frequency_filter(
            harmonic,
            src_param
        )
            
        # Generate noise component
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
        # Update the module's checkpointing flag
        if hasattr(self.sing_vocoder, 'use_checkpoint'):
            self.sing_vocoder.use_checkpoint = enabled
            
        return self