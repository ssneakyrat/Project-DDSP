import torch
import torch.nn as nn

from ddsp.melception import Melception
from ddsp.mel2control import Mel2Control
from ddsp.modules import HarmonicOscillator
from ddsp.core import scale_function, unit_to_hz2, frequency_filter, upsample

class Synth(nn.Module):
    def __init__(self, 
            sampling_rate,
            block_size,
            n_mag_harmonic,
            n_mag_noise,
            n_harmonics,
            n_mels=80):
        super().__init__()

        # params
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        
        # Mel2Control
        split_map = {
            'f0': 1, 
            'A': 1,
            'amplitudes': n_harmonics,
            'harmonic_magnitude': n_mag_harmonic,
            'noise_magnitude': n_mag_noise
        }
        self.mel2ctrl = Melception(n_mels, split_map)

        # Harmonic Synthsizer
        self.harmonic_synthsizer = HarmonicOscillator(sampling_rate)

    def forward(self, mel, initial_phase=None):
        '''
            mel: B x n_frames x n_mels
        '''

        ctrls = self.mel2ctrl(mel)

        # unpack
        f0_unit = ctrls['f0']# units
        f0_unit = torch.sigmoid(f0_unit)
        f0 = unit_to_hz2(f0_unit, hz_min = 80.0, hz_max=1000.0)
        f0[f0<80] = 0

        pitch = f0
        
        A           = scale_function(ctrls['A'])
        amplitudes  = scale_function(ctrls['amplitudes'])
        src_param = scale_function(ctrls['harmonic_magnitude'])
        noise_param = scale_function(ctrls['noise_magnitude'])

        amplitudes /= amplitudes.sum(-1, keepdim=True) # to distribution
        amplitudes *= A

        # exciter signal
        B, n_frames, _ = pitch.shape
        
        # upsample
        pitch = upsample(pitch, self.block_size)
        amplitudes = upsample(amplitudes, self.block_size)

        # harmonic
        harmonic, final_phase = self.harmonic_synthsizer(
            pitch, amplitudes, initial_phase)
        harmonic = frequency_filter(
                        harmonic,
                        src_param)
            
        # noise part
        noise = torch.rand_like(harmonic).to(noise_param) * 2 - 1
        noise = frequency_filter(
                        noise,
                        noise_param)
        signal = harmonic + noise

        return signal, f0, final_phase, (harmonic, noise)