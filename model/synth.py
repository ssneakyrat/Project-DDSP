import torch
import torch.nn as nn
import torchaudio

# Import the optimized modules
from ddsp.modules import HarmonicOscillator
from model.sing_vocoder import SingVocoder

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

        self.harmonic_synthsizer = HarmonicOscillator(sampling_rate)

        # Add mel extractor for unified mel extraction
        self.mel_extractor = MelExtractor(
            sample_rate=sampling_rate,
            hop_length=block_size,
            n_mels=n_mels
        )
        
        # Add mel refinement network
        self.mel_refiner = MelRefinementNetwork(n_mels=n_mels)

    def forward(self, batch, initial_phase=None, guidance_ratio=1.0):
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
        
        # Save the original frame-rate amplitudes for loss calculation
        frame_rate_amplitudes = amplitudes.clone()

        # Upsample to audio rate
        pitch = upsample(pitch, self.block_size)
        amplitudes = upsample(amplitudes, self.block_size)

        # harmonic
        harmonic, final_phase = self.harmonic_synthsizer(
            pitch, amplitudes, initial_phase)
        harmonic = frequency_filter(
                        harmonic,
                        src_param)

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

        # Extract mel from synthesized signal (unified mel extraction)
        signal_mel = self.mel_extractor(signal)
        
        # Apply mel refinement if guidance_ratio > 0 and target mel is available
        if guidance_ratio > 0 and 'mel' in batch:
            # Ensure target mel has the right format [B, mels, time]
            target_mel = batch['mel']
            if target_mel.shape[-1] != self.mel_extractor.mel_transform.n_mels:
                # Input is [batch, time, mels] but we need [batch, mels, time]
                target_mel = target_mel.transpose(1, 2)
                
            # Apply refinement
            signal = signal.unsqueeze(1)  # Add channel dimension [B, 1, T]
            signal = self.mel_refiner(signal, signal_mel, target_mel, guidance_ratio)
            signal = signal.squeeze(1)  # Remove channel dimension
            
            # Update mel spectrogram after refinement if needed (for loss calculation)
            if guidance_ratio > 0.3:  # Only recompute if significant refinement was applied
                signal_mel = self.mel_extractor(signal)

        return signal, f0, final_phase, (harmonic, noise, frame_rate_amplitudes, signal_mel)
    
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
    
class MelExtractor(nn.Module):
    """Extracts mel spectrograms from audio signals."""
    def __init__(self, sample_rate, n_fft=1024, hop_length=256, n_mels=80):
        super().__init__()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        
    def forward(self, audio):
        """
        Args:
            audio: Audio waveform [batch_size, audio_length]
        Returns:
            mel: Mel spectrogram [batch_size, n_mels, time_frames]
        """
        # Compute mel spectrogram
        mel = self.mel_transform(audio)
        
        # Convert to log-mel scale
        mel = torch.log(torch.clamp(mel, min=1e-5))
        
        return mel

class MelRefinementNetwork(nn.Module):
    """Refines audio based on mel spectrogram guidance."""
    def __init__(self, n_mels=80):
        super().__init__()
        
        # Convolutional refinement network
        self.audio_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.LeakyReLU(0.1),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.LeakyReLU(0.1),
        )
        
        # Mel condition network (processes both current and target mels)
        self.mel_encoder = nn.Sequential(
            nn.Conv1d(n_mels, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
        )
        
        # Fusion module
        self.fusion = nn.Sequential(
            nn.Conv1d(64 + 64 + 64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(32, 1, kernel_size=7, padding=3),
        )
        
    def forward(self, audio, current_mel, target_mel, guidance_ratio=1.0):
        """Apply mel-guided refinement with dynamic masking
        
        Args:
            audio: Input audio signal [B, 1, T]
            current_mel: Current mel spectrogram [B, mels, T_mel]
            target_mel: Target mel spectrogram [B, mels, T_mel]
            guidance_ratio: Amount of guidance to apply (0.0 = no guidance, 1.0 = full guidance)
        """
        # Skip refinement if no guidance
        if guidance_ratio <= 0:
            return audio
            
        # Apply masking proportional to guidance_ratio
        # Higher guidance_ratio = less masking
        masking_ratio = 1.0 - guidance_ratio
        
        # Apply progressive masking to target mel
        masked_target_mel = self.apply_masking(target_mel, masking_ratio)
        
        # Encode audio
        audio_features = self.audio_encoder(audio)
        
        # Encode current and target mel specs (need to adapt time dimension)
        # Ensure mels have same time dimension by interpolating if needed
        if current_mel.shape[2] != audio_features.shape[2]:
            current_mel = nn.functional.interpolate(
                current_mel, 
                size=audio_features.shape[2],
                mode='linear', 
                align_corners=False
            )
            
        if masked_target_mel.shape[2] != audio_features.shape[2]:
            masked_target_mel = nn.functional.interpolate(
                masked_target_mel, 
                size=audio_features.shape[2],
                mode='linear', 
                align_corners=False
            )
        
        # Encode mel spectrograms
        current_mel_features = self.mel_encoder(current_mel)
        target_mel_features = self.mel_encoder(masked_target_mel)
        
        # Concatenate features
        fused_features = torch.cat([audio_features, current_mel_features, target_mel_features], dim=1)
        
        # Apply fusion network to get residual
        residual = self.fusion(fused_features)
        
        # Apply residual connection
        refined_audio = audio + residual
        
        return refined_audio
        
    def apply_masking(self, mel, masking_ratio):
        """Apply progressive masking to mel spectrograms"""
        if masking_ratio <= 0:
            return mel
            
        # Random masking of time frames
        batch_size, n_mels, n_frames = mel.shape
        time_mask = torch.rand(batch_size, 1, n_frames).to(mel.device) > masking_ratio
        
        # Frequency masking (more advanced)
        freq_mask = torch.rand(batch_size, n_mels, 1).to(mel.device) > masking_ratio/2
        
        # Combine masks (both dimensions need to pass to keep information)
        combined_mask = time_mask & freq_mask
        
        # Add noise proportional to masking ratio
        noise_scale = masking_ratio * 0.1
        noise = torch.randn_like(mel) * noise_scale
        
        # Return masked mel with noise
        return mel * combined_mask + noise