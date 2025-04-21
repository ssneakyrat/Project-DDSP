import os
import torch
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

from ddsp.svs_vocoder import SVSVocoder
from ddsp.svs_loss import SVSHybridLoss

def test_phoneme2control():
    """Test the Phoneme2Control module"""
    from ddsp.phoneme2control import Phoneme2Control
    
    # Define test parameters
    batch_size = 2
    seq_length = 10
    num_phonemes = 50
    num_singers = 5
    num_languages = 3
    hidden_dim = 64
    
    # Define test control splits
    output_splits = {
        'f0': 1,
        'A': 1,
        'amplitudes': 10,
        'harmonic_magnitude': 20,
        'noise_magnitude': 10
    }
    
    # Create test inputs
    phonemes = torch.randint(0, num_phonemes, (batch_size, seq_length))
    durations = torch.ones(batch_size, seq_length)
    f0 = torch.rand(batch_size, seq_length)
    singer_ids = torch.randint(0, num_singers, (batch_size,))
    language_ids = torch.randint(0, num_languages, (batch_size,))
    
    # Create model
    model = Phoneme2Control(
        num_phonemes=num_phonemes,
        num_singers=num_singers,
        num_languages=num_languages,
        output_splits=output_splits,
        hidden_dim=hidden_dim
    )
    
    # Forward pass
    controls, formant_params = model(phonemes, durations, f0, singer_ids, language_ids)
    
    # Check output shapes
    assert 'f0' in controls, "F0 missing from controls"
    assert 'A' in controls, "A missing from controls"
    assert 'amplitudes' in controls, "Amplitudes missing from controls"
    assert 'harmonic_magnitude' in controls, "Harmonic magnitude missing from controls"
    assert 'noise_magnitude' in controls, "Noise magnitude missing from controls"
    
    assert controls['f0'].shape == (batch_size, seq_length, 1), "F0 shape incorrect"
    assert controls['amplitudes'].shape == (batch_size, seq_length, 10), "Amplitudes shape incorrect"
    
    formant_freqs, formant_bws, formant_amps = formant_params
    assert formant_freqs.shape == (batch_size, seq_length, 5), "Formant frequencies shape incorrect"
    assert formant_bws.shape == (batch_size, seq_length, 5), "Formant bandwidths shape incorrect"
    assert formant_amps.shape == (batch_size, seq_length, 5), "Formant amplitudes shape incorrect"
    
    print("Phoneme2Control test passed!")

def test_formant_filter():
    """Test the FormantFilter module"""
    from ddsp.formant_filter import FormantFilter
    
    # Define test parameters
    batch_size = 2
    audio_length = 24000
    n_frames = 100
    n_formants = 5
    sampling_rate = 24000
    
    # Create test inputs
    audio = torch.rand(batch_size, audio_length)
    formant_freqs = torch.rand(batch_size, n_frames, n_formants) * 5000 + 100  # 100-5100 Hz
    formant_bws = torch.rand(batch_size, n_frames, n_formants) * 500 + 50      # 50-550 Hz
    formant_amps = torch.softmax(torch.rand(batch_size, n_frames, n_formants), dim=-1)
    
    # Create model
    model = FormantFilter(sampling_rate)
    
    # Forward pass
    filtered_audio = model(audio, formant_freqs, formant_bws, formant_amps)
    
    # Check output shape
    assert filtered_audio.shape == audio.shape, "Filtered audio shape doesn't match input audio shape"
    
    print("FormantFilter test passed!")

def test_svs_vocoder():
    """Test the SVSVocoder module"""
    # Define test parameters
    batch_size = 2
    seq_length = 10
    audio_length = 24000
    sampling_rate = 24000
    block_size = 240
    n_mag_harmonic = 64
    n_mag_noise = 32
    n_harmonics = 40
    num_phonemes = 50
    num_singers = 5
    num_languages = 3
    
    # Create test inputs
    phonemes = torch.randint(0, num_phonemes, (batch_size, seq_length))
    durations = torch.ones(batch_size, seq_length)
    f0 = torch.rand(batch_size, seq_length) * 500 + 100  # 100-600 Hz
    singer_ids = torch.randint(0, num_singers, (batch_size,))
    language_ids = torch.randint(0, num_languages, (batch_size,))
    
    # Create model
    model = SVSVocoder(
        sampling_rate=sampling_rate,
        block_size=block_size,
        n_mag_harmonic=n_mag_harmonic,
        n_mag_noise=n_mag_noise,
        n_harmonics=n_harmonics,
        num_phonemes=num_phonemes,
        num_singers=num_singers,
        num_languages=num_languages
    )
    
    # Forward pass
    signal, f0_pred, phase, components, formant_params = model(
        phonemes, durations, f0, singer_ids, language_ids
    )
    
    # Check output shapes
    harmonic, noise = components
    formant_freqs, formant_bws, formant_amps = formant_params
    
    # Since we're using block_size=240, and seq_length=10, we should get audio of length ~2400
    # (there may be some edge effects, so we allow some flexibility)
    assert signal.shape[1] >= 2000, "Signal length too short"
    assert harmonic.shape == signal.shape, "Harmonic component shape doesn't match signal shape"
    assert noise.shape == signal.shape, "Noise component shape doesn't match signal shape"
    
    # Save some example audio
    os.makedirs("test_output", exist_ok=True)
    sf.write("test_output/svs_signal.wav", signal[0].detach().cpu().numpy(), sampling_rate)
    sf.write("test_output/svs_harmonic.wav", harmonic[0].detach().cpu().numpy(), sampling_rate)
    sf.write("test_output/svs_noise.wav", noise[0].detach().cpu().numpy(), sampling_rate)
    
    # Plot formant params for visualization
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.imshow(formant_freqs[0].detach().cpu().numpy().T, aspect='auto')
    plt.title("Formant Frequencies")
    plt.ylabel("Formant Index")
    
    plt.subplot(3, 1, 2)
    plt.imshow(formant_bws[0].detach().cpu().numpy().T, aspect='auto')
    plt.title("Formant Bandwidths")
    plt.ylabel("Formant Index")
    
    plt.subplot(3, 1, 3)
    plt.imshow(formant_amps[0].detach().cpu().numpy().T, aspect='auto')
    plt.title("Formant Amplitudes")
    plt.ylabel("Formant Index")
    plt.xlabel("Frame Index")
    
    plt.tight_layout()
    plt.savefig("test_output/formant_params.png")
    
    print("SVSVocoder test passed!")
    
def test_svs_loss():
    """Test the SVSHybridLoss module"""
    # Define test parameters
    batch_size = 2
    audio_length = 24000
    f0_length = 100
    sampling_rate = 24000
    n_ffts = [1024, 512, 256, 128]
    
    # Create test inputs
    y_pred = torch.rand(batch_size, audio_length)
    y_true = torch.rand(batch_size, audio_length)
    f0_pred = torch.rand(batch_size, f0_length)
    f0_true = torch.rand(batch_size, f0_length)
    
    # Create model
    loss_func = SVSHybridLoss(n_ffts, sampling_rate)
    
    # Forward pass
    loss, (loss_mss, loss_f0, loss_mel) = loss_func(y_pred, y_true, f0_pred, f0_true)
    
    # Check output types
    assert isinstance(loss.item(), float), "Loss is not a scalar"
    assert isinstance(loss_mss.item(), float), "MSS loss is not a scalar"
    assert isinstance(loss_f0.item(), float), "F0 loss is not a scalar"
    assert isinstance(loss_mel.item(), float), "Mel loss is not a scalar"
    
    print("SVSHybridLoss test passed!")

if __name__ == "__main__":
    print("Running SVS implementation tests...")
    test_phoneme2control()
    test_formant_filter()
    test_svs_vocoder()
    test_svs_loss()
    print("All tests passed!")