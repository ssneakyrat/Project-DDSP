"""
Standalone GPU-Accelerated Vocal Synthesizer
-------------------------------------------
A high-performance implementation of the VocalOscillator that can be used
both within ML models during training and as a standalone component for inference.
"""

import torch
import torch.nn as nn
import numpy as np
import time
from enum import Enum
from typing import Dict, Tuple, Optional, Union, List


class PrecisionMode(Enum):
    """Supported precision modes for computation"""
    FLOAT32 = "float32"
    FLOAT16 = "float16"


class FormantQuality(Enum):
    """Quality settings for formant processing"""
    LOW = "low"       # Fewer formants, simplified model
    NORMAL = "normal" # Default quality
    HIGH = "high"     # More formants, higher resolution


class BreathQuality(Enum):
    """Quality settings for breath noise"""
    LOW = "low"       # Simple noise model
    NORMAL = "normal" # Default quality
    HIGH = "high"     # Complex spectral shaping


class VocalSynthConfig:
    """Configuration for the vocal synthesizer with adjustable quality settings"""
    def __init__(
            self,
            sampling_rate: int = 22050,
            n_formants: int = 4,
            precision: Union[PrecisionMode, str] = PrecisionMode.FLOAT32,
            formant_quality: Union[FormantQuality, str] = FormantQuality.NORMAL,
            breath_quality: Union[BreathQuality, str] = BreathQuality.NORMAL,
            use_gpu: bool = True,
            fallback_to_cpu: bool = True,
            default_vibrato_rate: float = 5.5,
            default_vibrato_depth: float = 0.1,
            default_breathiness: float = 0.1,
            allow_precompute: bool = True
    ):
        self.sampling_rate = sampling_rate
        self.n_formants = n_formants
        
        # Set precision mode
        if isinstance(precision, str):
            self.precision = PrecisionMode(precision)
        else:
            self.precision = precision
            
        # Set quality modes
        if isinstance(formant_quality, str):
            self.formant_quality = FormantQuality(formant_quality)
        else:
            self.formant_quality = formant_quality
            
        if isinstance(breath_quality, str):
            self.breath_quality = BreathQuality(breath_quality)
        else:
            self.breath_quality = breath_quality
        
        # Device settings
        self.use_gpu = use_gpu
        self.fallback_to_cpu = fallback_to_cpu
        
        # Default parameters
        self.default_vibrato_rate = default_vibrato_rate
        self.default_vibrato_depth = default_vibrato_depth
        self.default_breathiness = default_breathiness
        
        # Performance optimizations
        self.allow_precompute = allow_precompute
        
    def get_device(self) -> torch.device:
        """Determine the appropriate device based on configuration and availability"""
        if self.use_gpu and torch.cuda.is_available():
            return torch.device('cuda')
        elif self.fallback_to_cpu:
            return torch.device('cpu')
        else:
            raise RuntimeError("GPU requested but not available, and fallback to CPU disabled")
            
    def get_dtype(self) -> torch.dtype:
        """Get the appropriate dtype based on precision setting"""
        if self.precision == PrecisionMode.FLOAT16:
            return torch.float16
        else:
            return torch.float32
    
    def get_formant_count(self) -> int:
        """Get the number of formants based on quality setting"""
        if self.formant_quality == FormantQuality.LOW:
            return min(2, self.n_formants)
        elif self.formant_quality == FormantQuality.HIGH:
            return max(6, self.n_formants)
        else:
            return self.n_formants
    
    def get_breath_bands(self) -> int:
        """Get the number of breath bands based on quality setting"""
        if self.breath_quality == FormantQuality.LOW:
            return 4
        elif self.breath_quality == FormantQuality.HIGH:
            return 16
        else:
            return 8


class AbstractVocalSynthesizer:
    """Abstract interface for vocal synthesis implementations"""
    def __init__(self, config: VocalSynthConfig = None):
        if config is None:
            config = VocalSynthConfig()
        self.config = config
        
    def __call__(self, *args, **kwargs):
        """
        Make the class callable for backward compatibility with nn.Module interface
        
        This simply forwards the call to the synthesize method
        """
        return self.synthesize(*args, **kwargs)
        
    def synthesize(
            self,
            f0: torch.Tensor,
            amplitudes: torch.Tensor,
            vibrato_rate: Optional[torch.Tensor] = None,
            vibrato_depth: Optional[torch.Tensor] = None,
            formant_freqs: Optional[torch.Tensor] = None,
            formant_bandwidths: Optional[torch.Tensor] = None,
            formant_gains: Optional[torch.Tensor] = None,
            glottal_open_quotient: Optional[torch.Tensor] = None,
            phase_coherence: Optional[torch.Tensor] = None,
            breathiness: Optional[torch.Tensor] = None,
            breath_spectral_shape: Optional[torch.Tensor] = None,
            initial_phase: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Synthesize audio from parameters
        
        Args:
            f0: B x T x 1 (Hz)
            amplitudes: B x T x n_harmonic
            vibrato_rate: B x T x 1 (Hz) or None
            vibrato_depth: B x T x 1 (semitones) or None
            formant_freqs: B x T x n_formants or None
            formant_bandwidths: B x T x n_formants or None
            formant_gains: B x T x n_formants or None
            glottal_open_quotient: B x T x 1 or None
            phase_coherence: B x T x 1 or None
            breathiness: B x T x 1 or None
            breath_spectral_shape: B x T x n_bands or None
            initial_phase: B x 1 x 1 or None
            
        Returns:
            signal: B x T
            final_phase: B x 1 x 1
        """
        raise NotImplementedError()
    
    @staticmethod
    def create(sampling_rate: int = 22050, use_gpu: bool = True, config=None, **kwargs) -> 'AbstractVocalSynthesizer':
        """Factory method to create appropriate implementation"""
        if config is None:
            config = VocalSynthConfig(sampling_rate=sampling_rate, use_gpu=use_gpu, **kwargs)
        
        # Determine which implementation to use
        if config.use_gpu and torch.cuda.is_available():
            return OptimizedVocalSynthesizer(config)
        elif config.fallback_to_cpu:
            # Use the PyTorch implementation on CPU
            return OptimizedVocalSynthesizer(config)
        else:
            raise RuntimeError("GPU requested but not available, and fallback to CPU disabled")


class OptimizedVocalSynthesizer(AbstractVocalSynthesizer):
    """
    Optimized implementation of VocalOscillator using PyTorch with GPU optimizations.
    
    This implementation vectorizes critical operations and provides performance
    optimizations while maintaining the same behavior as the original.
    """
    def __call__(self, *args, **kwargs):
        """
        Make the class callable for backward compatibility with nn.Module interface
        
        This simply forwards the call to the synthesize method
        """
        return self.synthesize(*args, **kwargs)
    def __init__(self, config: VocalSynthConfig = None):
        super().__init__(config)
        
        # Store config
        self.sampling_rate = self.config.sampling_rate
        self.device = self.config.get_device()
        self.dtype = self.config.get_dtype()
        
        # Precomputed values for efficiency
        if self.config.allow_precompute:
            self._precompute_constants()
        
        # Default formant values (soprano)
        self.default_formant_freqs = torch.tensor(
            [800.0, 1150.0, 2900.0, 3900.0, 4500.0, 5200.0][:self.config.get_formant_count()],
            device=self.device, dtype=self.dtype
        )
        self.default_formant_bandwidths = torch.tensor(
            [80.0, 90.0, 120.0, 130.0, 150.0, 170.0][:self.config.get_formant_count()],
            device=self.device, dtype=self.dtype
        )
        self.default_formant_gains = torch.tensor(
            [1.0, 0.5, 0.25, 0.1, 0.05, 0.025][:self.config.get_formant_count()],
            device=self.device, dtype=self.dtype
        )
        
        # Set default parameters
        self.default_vibrato_rate = self.config.default_vibrato_rate
        self.default_vibrato_depth = self.config.default_vibrato_depth
        self.default_breathiness = self.config.default_breathiness
        self.default_glottal_open_quotient = 0.7
        
    def _precompute_constants(self):
        """Precompute constants used in synthesis for efficiency"""
        # Could precompute FFT kernels, window functions, etc.
        # This would be implementation-specific
        pass
    
    def synthesize(
            self,
            f0: torch.Tensor,
            amplitudes: torch.Tensor,
            vibrato_rate: Optional[torch.Tensor] = None,
            vibrato_depth: Optional[torch.Tensor] = None,
            formant_freqs: Optional[torch.Tensor] = None,
            formant_bandwidths: Optional[torch.Tensor] = None,
            formant_gains: Optional[torch.Tensor] = None,
            glottal_open_quotient: Optional[torch.Tensor] = None,
            phase_coherence: Optional[torch.Tensor] = None,
            breathiness: Optional[torch.Tensor] = None,
            breath_spectral_shape: Optional[torch.Tensor] = None,
            initial_phase: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Convert tensors to the target device and dtype
        f0 = self._prepare_tensor(f0)
        amplitudes = self._prepare_tensor(amplitudes)
        
        batch_size = f0.shape[0]
        time_steps = f0.shape[1]
        n_harmonics = amplitudes.shape[2]
        
        # Initialize default parameters if not provided
        if initial_phase is None:
            initial_phase = torch.zeros(batch_size, 1, 1, device=self.device, dtype=self.dtype)
        else:
            initial_phase = self._prepare_tensor(initial_phase)

        # Vibrato parameters
        if vibrato_rate is None:
            vibrato_rate = torch.ones(batch_size, 1, 1, device=self.device, dtype=self.dtype) * self.default_vibrato_rate
        else:
            vibrato_rate = self._prepare_tensor(vibrato_rate)
            
        if vibrato_depth is None:
            vibrato_depth = torch.ones(batch_size, 1, 1, device=self.device, dtype=self.dtype) * self.default_vibrato_depth
        else:
            vibrato_depth = self._prepare_tensor(vibrato_depth)

        # Formant parameters
        n_formants = self.default_formant_freqs.size(0)
        if formant_freqs is None:
            formant_freqs = self.default_formant_freqs.reshape(1, 1, n_formants).repeat(batch_size, time_steps, 1)
        else:
            formant_freqs = self._prepare_tensor(formant_freqs)
            
        if formant_bandwidths is None:
            formant_bandwidths = self.default_formant_bandwidths.reshape(1, 1, n_formants).repeat(batch_size, time_steps, 1)
        else:
            formant_bandwidths = self._prepare_tensor(formant_bandwidths)
            
        if formant_gains is None:
            formant_gains = self.default_formant_gains.reshape(1, 1, n_formants).repeat(batch_size, time_steps, 1)
        else:
            formant_gains = self._prepare_tensor(formant_gains)

        # Glottal model parameters
        if glottal_open_quotient is None:
            glottal_open_quotient = torch.ones(batch_size, time_steps, 1, device=self.device, dtype=self.dtype) * self.default_glottal_open_quotient
        else:
            glottal_open_quotient = self._prepare_tensor(glottal_open_quotient)
            
        if phase_coherence is None:
            phase_coherence = torch.ones(batch_size, time_steps, 1, device=self.device, dtype=self.dtype) * 0.5  # Default 50% coherence
        else:
            phase_coherence = self._prepare_tensor(phase_coherence)

        # Breathiness parameters
        if breathiness is None:
            breathiness = torch.ones(batch_size, time_steps, 1, device=self.device, dtype=self.dtype) * self.default_breathiness
        else:
            breathiness = self._prepare_tensor(breathiness)
            
        if breath_spectral_shape is None:
            # Default: decreasing energy at higher frequencies
            n_bands = self.config.get_breath_bands()
            breath_spectral_shape = torch.pow(
                0.7, 
                torch.arange(n_bands, device=self.device, dtype=self.dtype).reshape(1, 1, -1).repeat(batch_size, time_steps, 1)
            )
        else:
            breath_spectral_shape = self._prepare_tensor(breath_spectral_shape)
        
        # Get voiced mask
        mask = (f0 > 0).detach()
        
        # 1. Apply vibrato to f0
        f0_vibrato = self._apply_vibrato_optimized(f0, vibrato_rate, vibrato_depth)

        # 2. Calculate phase
        phase = torch.cumsum(2 * torch.pi * f0_vibrato / self.sampling_rate, dim=1) + initial_phase
        
        # Generate harmonic phases - vectorized version
        harmonic_indices = torch.arange(1, n_harmonics + 1, device=self.device, dtype=self.dtype)
        phases = phase * harmonic_indices.reshape(1, 1, -1)
        
        # 3. Apply phase coherence
        phases = self._apply_phase_coherence_optimized(phases, phase_coherence)
        
        # 4. Apply anti-aliasing if needed
        amp = self._remove_above_nyquist(amplitudes, f0_vibrato)
        
        # 5. Use glottal flow instead of sinusoid
        excitation = self._glottal_flow_model_optimized(phases, glottal_open_quotient)
        harmonic = (excitation * amp).sum(-1)
        
        # 6. Apply formant filtering - vectorized version
        harmonic = self._apply_formant_filter_vectorized(harmonic, formant_freqs, formant_bandwidths, formant_gains)
        
        # 7. Generate and add breath noise - vectorized version
        breath_noise = self._generate_breath_noise_optimized(
            (batch_size, time_steps), 
            breath_spectral_shape
        )
        
        # Scale noise by breathiness
        scaled_noise = breath_noise * breathiness.squeeze(-1)
        
        # 8. Mix harmonic and noise components
        # Apply voiced mask to harmonic
        harmonic = harmonic * mask.squeeze(-1)
        
        # Final signal (mix harmonic and breath components)
        signal = harmonic + scaled_noise

        # Return final phase for continuity
        final_phase = phase[:, -1:, :] % (2 * torch.pi)
        
        return signal, final_phase.detach()
    
    def _prepare_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is on the right device and has the right dtype"""
        return tensor.to(device=self.device, dtype=self.dtype)
    
    def _apply_vibrato_optimized(self, f0, vibrato_rate, vibrato_depth):
        """
        Vectorized implementation of vibrato application
        
        Args:
            f0: B x T x 1 tensor with fundamental frequencies in Hz
            vibrato_rate: B x T x 1 tensor with vibrato rates in Hz
            vibrato_depth: B x T x 1 tensor with vibrato depth in semitones
            
        Returns:
            f0_vibrato: B x T x 1 tensor with modulated frequencies
        """
        batch_size, time_steps, _ = f0.shape
        
        # Create time vector - fully vectorized
        time = torch.arange(0, time_steps, device=self.device, dtype=self.dtype) / self.sampling_rate
        time = time.reshape(1, -1, 1).expand(batch_size, -1, 1)
        
        # Apply sinusoidal vibrato
        # Convert semitones to multiplication factor: 2^(depth/12)
        vibrato_factor = 2.0 ** (vibrato_depth / 12.0)
        
        # Modulating between 1/factor and factor using sine wave
        # This computation is fully vectorized now
        vibrato_multiplier = torch.exp(
            torch.sin(2 * torch.pi * vibrato_rate * time) * 
            torch.log(vibrato_factor)
        )
        
        # Apply to f0, only where f0 > 0 (voiced regions)
        f0_vibrato = f0 * vibrato_multiplier
        f0_vibrato = torch.where(f0 > 0, f0_vibrato, f0)
        
        return f0_vibrato
    
    def _glottal_flow_model_optimized(self, phase, open_quotient):
        """
        Vectorized glottal flow model
        
        Args:
            phase: B x T x n_harmonic tensor with phases for each harmonic
            open_quotient: B x T x 1 tensor with open quotient (0.0-1.0)
            
        Returns:
            flow: B x T x n_harmonic tensor with glottal flow waveforms
        """
        # Normalize phase to [0, 1] range
        norm_phase = (phase % (2 * torch.pi)) / (2 * torch.pi)
        
        # Expand open_quotient for broadcasting
        open_quotient_expanded = open_quotient.expand_as(norm_phase)
        
        # Rising phase (open phase) - vectorized
        rising = norm_phase / open_quotient_expanded
        rising_flow = torch.sin(rising * torch.pi / 2.0) ** 2
        rising_mask = (norm_phase < open_quotient_expanded).float()
        
        # Falling phase (return phase) - vectorized
        falling = (norm_phase - open_quotient_expanded) / (1.0 - open_quotient_expanded + 1e-7)  # Avoid division by zero
        falling_flow = torch.cos(falling * torch.pi / 2.0) ** 2
        falling_mask = (norm_phase >= open_quotient_expanded).float()
        
        # Combine phases - vectorized
        flow = rising_flow * rising_mask + falling_flow * falling_mask
        
        # Adjust DC offset to ensure mean is 0
        flow = flow - 0.5
        
        return flow
    
    def _apply_phase_coherence_optimized(self, phases, coherence_factor):
        """
        Vectorized phase coherence adjustment
        
        Args:
            phases: B x T x n_harmonic tensor with phases
            coherence_factor: B x T x 1 tensor with coherence factor (0.0-1.0)
            
        Returns:
            modified_phases: B x T x n_harmonic tensor with adjusted phases
        """
        # Extract fundamental phase
        fundamental_phase = phases[:, :, 0:1]
        
        # For higher harmonics, blend between independent phase and 
        # phase that would be coherent with fundamental
        if phases.size(2) > 1:
            # This is now fully vectorized
            harmonic_indices = torch.arange(1, phases.size(2), device=self.device, dtype=self.dtype)
            harmonic_indices = harmonic_indices.reshape(1, 1, -1)
            
            coherent_phases = fundamental_phase * harmonic_indices
            
            # Apply coherence blending to harmonics 2+ - vectorized
            higher_harmonics = phases[:, :, 1:]
            
            # Expand coherence_factor for broadcasting
            coherence_factor_expanded = coherence_factor.expand(-1, -1, phases.size(2) - 1)
            
            blended_phases = (
                coherence_factor_expanded * coherent_phases + 
                (1 - coherence_factor_expanded) * higher_harmonics
            )
            
            # Recombine with fundamental
            modified_phases = torch.cat([fundamental_phase, blended_phases], dim=2)
        else:
            modified_phases = phases
            
        return modified_phases
    
    def _apply_formant_filter_vectorized(self, signal, formant_freqs, formant_bandwidths, formant_gains):
        """
        Vectorized formant filtering using frequency domain processing
        
        Args:
            signal: B x T tensor with input signal
            formant_freqs: B x T x n_formants tensor with formant frequencies
            formant_bandwidths: B x T x n_formants tensor with formant bandwidths
            formant_gains: B x T x n_formants tensor with formant gains
            
        Returns:
            filtered_signal: B x T tensor with formant-filtered signal
        """
        batch_size, time_steps = signal.shape
        
        # Get frequency domain representation
        signal_fft = torch.fft.rfft(signal)
        freqs = torch.fft.rfftfreq(time_steps, d=1.0/self.sampling_rate).to(signal.device)
        num_freqs = len(freqs)
        
        # Extract formant parameters at middle frame (for simplicity)
        mid_frame = formant_freqs.shape[1] // 2
        formant_freqs_mid = formant_freqs[:, mid_frame, :]        # B x n_formants
        formant_bandwidths_mid = formant_bandwidths[:, mid_frame, :]  # B x n_formants
        formant_gains_mid = formant_gains[:, mid_frame, :]        # B x n_formants
        n_formants = formant_freqs_mid.shape[1]
        
        # Reshape for broadcasting
        freqs_reshaped = freqs.reshape(1, -1, 1)  # 1 x F x 1
        formant_freqs_reshaped = formant_freqs_mid.reshape(batch_size, 1, n_formants)  # B x 1 x n_formants
        formant_bandwidths_reshaped = formant_bandwidths_mid.reshape(batch_size, 1, n_formants)  # B x 1 x n_formants
        formant_gains_reshaped = formant_gains_mid.reshape(batch_size, 1, n_formants)  # B x 1 x n_formants
        
        # Calculate response for all formants at once - fully vectorized
        response = formant_gains_reshaped * torch.pow(
            1.0 / (1.0 + torch.pow((freqs_reshaped - formant_freqs_reshaped) / (formant_bandwidths_reshaped/2.0 + 1e-7), 2)), 
            1.0
        )  # B x F x n_formants
        
        # Sum across formants
        freq_response = response.sum(dim=2)  # B x F
        
        # Apply filter
        filtered_fft = signal_fft * freq_response
        
        # Convert back to time domain
        filtered_signal = torch.fft.irfft(filtered_fft, n=time_steps)
        
        return filtered_signal
    
    def _generate_breath_noise_optimized(self, shape, spectral_shape):
        """
        Vectorized breath noise generation
        
        Args:
            shape: tuple with shape of noise to generate (B, T)
            spectral_shape: B x T x n_bands tensor controlling spectral shape
            
        Returns:
            breath_noise: B x T tensor with shaped noise
        """
        batch_size, time_steps = shape
        device = spectral_shape.device
        
        # Generate white noise
        noise = torch.randn(shape, device=device, dtype=self.dtype)
        
        # Extract spectral shape at middle frame
        mid_frame = spectral_shape.shape[1] // 2
        spectral_weights = spectral_shape[:, mid_frame, :]  # B x n_bands
        n_bands = spectral_weights.shape[1]
        
        # This is now fully vectorized
        shaped_noise = torch.zeros_like(noise)
        
        # Create a bank of smoothed noise with different characteristics
        # Create kernels of different sizes for different frequency bands
        kernel_sizes = []
        all_kernels = []
        
        for i in range(n_bands):
            # Smoothing factor determines frequency content
            smooth_factor = 2.0 ** (i / 2.0)  # Exponential scaling of smoothing
            
            # Create smoothed noise for this band using efficient convolution
            kernel_size = max(3, int(128 / smooth_factor))
            if kernel_size % 2 == 0:
                kernel_size += 1  # Ensure odd kernel size
                
            kernel_sizes.append(kernel_size)
            
            # Simple Gaussian-like kernel for smoothing
            kernel = torch.exp(-torch.arange(-(kernel_size//2), kernel_size//2 + 1, device=device, dtype=self.dtype)**2 / (2 * (kernel_size/6)**2))
            kernel = kernel / kernel.sum()  # Normalize
            all_kernels.append(kernel.view(1, 1, -1))
        
        # Process each band - we still need a loop here for different kernel sizes
        # but the internal operations are vectorized
        for i in range(n_bands):
            # Reshape for batch 1D convolution
            kernel = all_kernels[i]
            band_noise = noise.unsqueeze(1)  # B x 1 x T
            
            # Apply convolution for smoothing
            smoothed_noise = torch.nn.functional.conv1d(
                band_noise, 
                kernel, 
                padding='same'
            ).squeeze(1)  # Back to B x T
            
            # Weight by spectral shape and add to result
            band_weights = spectral_weights[:, i].view(batch_size, 1)
            shaped_noise = shaped_noise + smoothed_noise * band_weights
        
        # Normalize
        mean = shaped_noise.mean(dim=1, keepdim=True)
        std = shaped_noise.std(dim=1, keepdim=True) + 1e-8
        normalized_noise = (shaped_noise - mean) / std
        
        return normalized_noise
    
    def _remove_above_nyquist(self, amplitudes, f0):
        """
        Vectorized anti-aliasing filter
        
        Args:
            amplitudes: B x T x n_harmonic tensor with harmonic amplitudes
            f0: B x T x 1 tensor with fundamental frequencies in Hz
            
        Returns:
            filtered_amplitudes: B x T x n_harmonic tensor with filtered amplitudes
        """
        batch_size, n_frames, n_harmonics = amplitudes.shape
        
        # Create harmonic frequencies
        harmonic_numbers = torch.arange(1, n_harmonics + 1, device=self.device, dtype=self.dtype).reshape(1, 1, -1)
        harmonic_freqs = f0 * harmonic_numbers  # B x T x n_harmonics
        
        # Create Nyquist mask
        nyquist = self.sampling_rate / 2.0
        nyquist_mask = (harmonic_freqs < nyquist).float()
        
        # Apply mask
        return amplitudes * nyquist_mask


# Advanced implementation with CUDA kernels for critical operations
class CUDAVocalSynthesizer(OptimizedVocalSynthesizer):
    """
    High-performance implementation using custom CUDA kernels for the most
    computationally intensive operations.
    """
    def __init__(self, config: VocalSynthConfig = None):
        super().__init__(config)
        
        # Check if we can use CUDA
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available, cannot use CUDAVocalSynthesizer")
        
        # Load CUDA kernels
        self._load_cuda_kernels()
        
        # Flag to control whether to use custom kernels or PyTorch
        self.use_custom_kernels = True
        
    def _load_cuda_kernels(self):
        """Load and compile CUDA kernels"""
        try:
            from torch.utils.cpp_extension import load_inline
            
            # Define CUDA kernels for critical operations
            
            # 1. Vibrato kernel - faster implementation of _apply_vibrato
            vibrato_cuda_source = """
            #include <torch/extension.h>
            #include <cuda.h>
            #include <cuda_runtime.h>
            #include <math.h>
            
            __global__ void vibrato_kernel(
                float* f0, float* vibrato_rate, float* vibrato_depth,
                float* output, int batch_size, int time_steps, float sampling_rate
            ) {
                // Calculate global position
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx >= batch_size * time_steps) return;
                
                int b = idx / time_steps;
                int t = idx % time_steps;
                
                // Get parameters
                float f0_val = f0[b * time_steps + t];
                float rate = vibrato_rate[b];
                float depth = vibrato_depth[b];
                
                // Skip unvoiced
                if (f0_val <= 0.0f) {
                    output[b * time_steps + t] = f0_val;
                    return;
                }
                
                // Calculate time
                float time = t / sampling_rate;
                
                // Calculate vibrato factor (2^(depth/12))
                float vibrato_factor = powf(2.0f, depth / 12.0f);
                
                // Calculate multiplier
                float angle = 2.0f * M_PI * rate * time;
                float vibrato_multiplier = expf(sinf(angle) * logf(vibrato_factor));
                
                // Apply
                output[b * time_steps + t] = f0_val * vibrato_multiplier;
            }
            
            torch::Tensor vibrato_cuda(
                torch::Tensor f0,
                torch::Tensor vibrato_rate,
                torch::Tensor vibrato_depth,
                float sampling_rate
            ) {
                // Get dimensions
                int batch_size = f0.size(0);
                int time_steps = f0.size(1);
                
                // Create output tensor
                auto output = torch::zeros_like(f0);
                
                // Calculate grid and block dimensions
                const int threads = 256;
                const int blocks = (batch_size * time_steps + threads - 1) / threads;
                
                // Launch kernel
                vibrato_kernel<<<blocks, threads>>>(
                    f0.data_ptr<float>(),
                    vibrato_rate.data_ptr<float>(),
                    vibrato_depth.data_ptr<float>(),
                    output.data_ptr<float>(),
                    batch_size, time_steps, sampling_rate
                );
                
                return output;
            }
            """
            
            # 2. Formant filtering kernel - optimized version of _apply_formant_filter
            formant_cuda_source = """
            #include <torch/extension.h>
            #include <cuda.h>
            #include <cuda_runtime.h>
            #include <math.h>
            
            __global__ void formant_response_kernel(
                float* freqs, float* formant_freqs, float* formant_bw, float* formant_gains,
                float* output, int batch_size, int n_freqs, int n_formants
            ) {
                // Calculate global position
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx >= batch_size * n_freqs) return;
                
                int b = idx / n_freqs;
                int f = idx % n_freqs;
                
                // Calculate response for each formant
                float response = 0.0f;
                float freq = freqs[f];
                
                for (int i = 0; i < n_formants; i++) {
                    float f0 = formant_freqs[b * n_formants + i];
                    float bw = formant_bw[b * n_formants + i];
                    float gain = formant_gains[b * n_formants + i];
                    
                    // Calculate response for this formant
                    float diff = freq - f0;
                    float bw_half = bw / 2.0f;
                    response += gain / (1.0f + (diff / bw_half) * (diff / bw_half));
                }
                
                // Store result
                output[b * n_freqs + f] = response;
            }
            
            torch::Tensor formant_response_cuda(
                torch::Tensor freqs,
                torch::Tensor formant_freqs,
                torch::Tensor formant_bandwidths,
                torch::Tensor formant_gains
            ) {
                // Get dimensions
                int batch_size = formant_freqs.size(0);
                int n_freqs = freqs.size(0);
                int n_formants = formant_freqs.size(1);
                
                // Create output tensor
                auto output = torch::zeros({batch_size, n_freqs}, freqs.options());
                
                // Calculate grid and block dimensions
                const int threads = 256;
                const int blocks = (batch_size * n_freqs + threads - 1) / threads;
                
                // Launch kernel
                formant_response_kernel<<<blocks, threads>>>(
                    freqs.data_ptr<float>(),
                    formant_freqs.data_ptr<float>(),
                    formant_bandwidths.data_ptr<float>(),
                    formant_gains.data_ptr<float>(),
                    output.data_ptr<float>(),
                    batch_size, n_freqs, n_formants
                );
                
                return output;
            }
            """
            
            # 3. Glottal flow kernel - optimized version of _glottal_flow_model
            glottal_cuda_source = """
            #include <torch/extension.h>
            #include <cuda.h>
            #include <cuda_runtime.h>
            #include <math.h>
            
            __global__ void glottal_flow_kernel(
                float* phase, float* open_quotient, float* output,
                int batch_size, int time_steps, int n_harmonics
            ) {
                // Calculate global position
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx >= batch_size * time_steps * n_harmonics) return;
                
                int b = idx / (time_steps * n_harmonics);
                int t = (idx % (time_steps * n_harmonics)) / n_harmonics;
                int h = idx % n_harmonics;
                
                // Get parameters
                float phase_val = phase[b * time_steps * n_harmonics + t * n_harmonics + h];
                float oq = open_quotient[b * time_steps + t];
                
                // Normalize phase to [0, 1] range
                float norm_phase = fmodf(phase_val, 2.0f * M_PI) / (2.0f * M_PI);
                
                // Calculate flow based on phase
                float flow;
                if (norm_phase < oq) {
                    // Rising phase (open phase)
                    float rising = norm_phase / oq;
                    flow = powf(sinf(rising * M_PI / 2.0f), 2);
                } else {
                    // Falling phase (return phase)
                    float falling = (norm_phase - oq) / (1.0f - oq);
                    flow = powf(cosf(falling * M_PI / 2.0f), 2);
                }
                
                // Adjust DC offset
                flow = flow - 0.5f;
                
                // Store result
                output[b * time_steps * n_harmonics + t * n_harmonics + h] = flow;
            }
            
            torch::Tensor glottal_flow_cuda(
                torch::Tensor phase,
                torch::Tensor open_quotient
            ) {
                // Get dimensions
                int batch_size = phase.size(0);
                int time_steps = phase.size(1);
                int n_harmonics = phase.size(2);
                
                // Create output tensor
                auto output = torch::zeros_like(phase);
                
                // Calculate grid and block dimensions
                const int threads = 256;
                const int blocks = (batch_size * time_steps * n_harmonics + threads - 1) / threads;
                
                // Launch kernel
                glottal_flow_kernel<<<blocks, threads>>>(
                    phase.data_ptr<float>(),
                    open_quotient.data_ptr<float>(),
                    output.data_ptr<float>(),
                    batch_size, time_steps, n_harmonics
                );
                
                return output;
            }
            """
            
            # Combine all sources
            cuda_source = vibrato_cuda_source + formant_cuda_source + glottal_cuda_source
            
            # Define Python interface
            python_source = """
            import torch
            
            def apply_vibrato_cuda(f0, vibrato_rate, vibrato_depth, sampling_rate):
                return vibrato_cuda(f0, vibrato_rate, vibrato_depth, sampling_rate)
                
            def formant_response_cuda(freqs, formant_freqs, formant_bandwidths, formant_gains):
                return formant_response_cuda(freqs, formant_freqs, formant_bandwidths, formant_gains)
                
            def glottal_flow_cuda(phase, open_quotient):
                return glottal_flow_cuda(phase, open_quotient)
            """
            
            # Compile extensions
            self.cuda_ext = load_inline(
                name="vocal_synth_cuda",
                cpp_sources="",
                cuda_sources=cuda_source,
                functions=["vibrato_cuda", "formant_response_cuda", "glottal_flow_cuda"],
                extra_cuda_cflags=["-O3"],
                with_cuda=True
            )
            
            # Register functions
            self.apply_vibrato_cuda = self.cuda_ext.vibrato_cuda
            self.formant_response_cuda = self.cuda_ext.formant_response_cuda
            self.glottal_flow_cuda = self.cuda_ext.glottal_flow_cuda
            
        except Exception as e:
            print(f"Warning: Failed to load CUDA extensions: {e}")
            print("Falling back to PyTorch implementation")
            self.use_custom_kernels = False
    
    def _apply_vibrato_optimized(self, f0, vibrato_rate, vibrato_depth):
        """
        GPU-accelerated vibrato implementation
        """
        if self.use_custom_kernels:
            try:
                # Convert to float32 for CUDA kernels
                f0_float32 = f0.float()
                vibrato_rate_float32 = vibrato_rate.float()
                vibrato_depth_float32 = vibrato_depth.float()
                
                # Call CUDA kernel
                result = self.apply_vibrato_cuda(
                    f0_float32,
                    vibrato_rate_float32,
                    vibrato_depth_float32,
                    float(self.sampling_rate)
                )
                
                # Convert back to original dtype
                return result.to(self.dtype)
            except Exception:
                # Fall back to PyTorch implementation
                return super()._apply_vibrato_optimized(f0, vibrato_rate, vibrato_depth)
        else:
            # Use PyTorch implementation
            return super()._apply_vibrato_optimized(f0, vibrato_rate, vibrato_depth)
    
    def _apply_formant_filter_vectorized(self, signal, formant_freqs, formant_bandwidths, formant_gains):
        """
        GPU-accelerated formant filtering
        """
        if self.use_custom_kernels:
            try:
                batch_size, time_steps = signal.shape
                
                # Get frequency domain representation
                signal_fft = torch.fft.rfft(signal)
                freqs = torch.fft.rfftfreq(time_steps, d=1.0/self.sampling_rate).to(signal.device)
                
                # Extract formant parameters at middle frame
                mid_frame = formant_freqs.shape[1] // 2
                formant_freqs_mid = formant_freqs[:, mid_frame, :]
                formant_bandwidths_mid = formant_bandwidths[:, mid_frame, :]
                formant_gains_mid = formant_gains[:, mid_frame, :]
                
                # Convert to float32 for CUDA kernels
                freqs_float32 = freqs.float()
                formant_freqs_float32 = formant_freqs_mid.float()
                formant_bandwidths_float32 = formant_bandwidths_mid.float()
                formant_gains_float32 = formant_gains_mid.float()
                
                # Call CUDA kernel to compute response
                freq_response = self.formant_response_cuda(
                    freqs_float32,
                    formant_freqs_float32,
                    formant_bandwidths_float32,
                    formant_gains_float32
                )
                
                # Convert back to original dtype
                freq_response = freq_response.to(self.dtype)
                
                # Apply filter
                filtered_fft = signal_fft * freq_response
                
                # Convert back to time domain
                return torch.fft.irfft(filtered_fft, n=time_steps)
            except Exception:
                # Fall back to PyTorch implementation
                return super()._apply_formant_filter_vectorized(signal, formant_freqs, formant_bandwidths, formant_gains)
        else:
            # Use PyTorch implementation
            return super()._apply_formant_filter_vectorized(signal, formant_freqs, formant_bandwidths, formant_gains)
    
    def _glottal_flow_model_optimized(self, phase, open_quotient):
        """
        GPU-accelerated glottal flow model
        """
        if self.use_custom_kernels:
            try:
                # Convert to float32 for CUDA kernels
                phase_float32 = phase.float()
                open_quotient_float32 = open_quotient.expand_as(phase).float()
                
                # Call CUDA kernel
                result = self.glottal_flow_cuda(
                    phase_float32,
                    open_quotient_float32
                )
                
                # Convert back to original dtype
                return result.to(self.dtype)
            except Exception:
                # Fall back to PyTorch implementation
                return super()._glottal_flow_model_optimized(phase, open_quotient)
        else:
            # Use PyTorch implementation
            return super()._glottal_flow_model_optimized(phase, open_quotient)

# Utility functions for benchmarking
def benchmark_synthesizer(synth, batch_size=1, time_steps=44100, n_harmonics=100):
    """
    Benchmark the synthesizer performance
    
    Args:
        synth: Synthesizer instance
        batch_size: Batch size to use
        time_steps: Number of time steps
        n_harmonics: Number of harmonics
        
    Returns:
        avg_time: Average time per synthesis in seconds
    """
    device = next(synth.parameters(), torch.zeros(1)).device if isinstance(synth, nn.Module) else synth.device
    
    # Create test inputs
    f0 = torch.ones((batch_size, time_steps, 1), device=device) * 220.0
    amplitudes = torch.ones((batch_size, time_steps, n_harmonics), device=device) / n_harmonics
    
    # Warm up
    for _ in range(3):
        with torch.no_grad():
            synth.synthesize(f0, amplitudes)
    
    # Benchmark
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    n_runs = 10
    for _ in range(n_runs):
        with torch.no_grad():
            synth.synthesize(f0, amplitudes)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    
    avg_time = (end_time - start_time) / n_runs
    return avg_time

# Example usage:
if __name__ == "__main__":
    # Create a synthesizer
    synth = AbstractVocalSynthesizer.create(sampling_rate=22050, use_gpu=True)
    
    # Benchmark
    print(f"Running benchmark...")
    avg_time = benchmark_synthesizer(synth)
    print(f"Average synthesis time: {avg_time:.4f} seconds")
    
    # Test with some basic parameters
    print("Testing synthesizer...")
    batch_size = 1
    time_steps = 22050  # 1 second of audio at 22050 Hz
    n_harmonics = 100
    
    # Create test inputs
    device = synth.device
    f0 = torch.ones((batch_size, time_steps, 1), device=device) * 220.0  # A3
    amplitudes = torch.zeros((batch_size, time_steps, n_harmonics), device=device)
    amplitudes[:, :, 0] = 0.5  # Fundamental
    amplitudes[:, :, 1] = 0.3  # 2nd harmonic
    amplitudes[:, :, 2] = 0.2  # 3rd harmonic
    
    # Synthesize
    signal, _ = synth.synthesize(f0, amplitudes)
    
    print(f"Generated audio shape: {signal.shape}")
    print("Done!")