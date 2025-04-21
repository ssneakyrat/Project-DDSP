import os
import sys
import time
import shutil
import numpy as np
import soundfile as sf
import librosa

import torch

from logger.saver import Saver
from logger import utils


def extract_mel_from_audio(audio, args):
    """
    Extract mel-spectrogram from audio signal using the same approach as in dataset.py.
    
    Args:
        audio: Audio signal (numpy array)
        args: Configuration arguments
    
    Returns:
        mel: Mel-spectrogram
    """
    import torch
    import torchaudio
    import numpy as np
    
    # Convert audio to tensor
    audio_tensor = torch.FloatTensor(audio).unsqueeze(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    audio_tensor = audio_tensor.to(device)
    
    # Initialize mel transform with the same parameters as in dataset.py
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=args.data.sampling_rate,
        n_fft=1024,
        win_length=1024,
        hop_length=args.data.block_size,
        f_min=40,  # FMIN from dataset.py
        f_max=12000,  # FMAX from dataset.py
        n_mels=80,
        power=2.0
    ).to(device)
    
    # Extract mel spectrogram
    mel_spec = mel_transform(audio_tensor)
    mel_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
    
    # Normalize mel spectrogram as in dataset.py
    mel_spec = mel_db - mel_db.min()
    mel_spec = mel_spec / (mel_spec.max() + 1e-8)
    
    # Move mel to CPU and convert to numpy with the same shape as in dataset.py
    mel_np = mel_spec.squeeze(0).cpu().numpy().T  # Transpose to get (time, n_mels)
    
    return mel_np

def inference_from_wav(args, model, path_wav_file, path_gendir='wav_gen', is_part=False):
    """
    Perform inference directly from a WAV file by splitting into 2-second chunks,
    processing each chunk separately with proper phase continuity.
    """
    print(' [*] Inferencing from WAV file using trained chunk parameters...')
    model.eval()
    
    # Import needed libraries
    try:
        import torchaudio
        import parselmouth
    except ImportError:
        print(" [!] torchaudio and parselmouth are required. Please install them.")
        return
    
    # Check if the input file exists
    if not os.path.exists(path_wav_file):
        print(f" [x] Input WAV file does not exist: {path_wav_file}")
        return
    
    # Extract filename from path
    fn = os.path.basename(path_wav_file).split('.')[0]
    print(f" > Processing file: {fn}")
    
    # Load WAV file
    audio, sr = sf.read(path_wav_file, dtype='float32')
    
    # Convert to mono if stereo
    if len(audio.shape) > 1 and audio.shape[1] > 1:
        print(" > Converting stereo to mono")
        audio = audio.mean(axis=1)

    # Resample if needed
    if sr != args.data.sampling_rate:
        # Use torchaudio for faster resampling
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0)
        resampler = torchaudio.transforms.Resample(
            orig_freq=sr, new_freq=args.data.sampling_rate
        )
        audio_tensor = resampler(audio_tensor)
        audio = audio_tensor.squeeze(0).numpy()
        sr = args.data.sampling_rate
    
    # Create directories
    chunks_dir = os.path.join(path_gendir, 'chunks')
    mel_dir = os.path.join(path_gendir, 'mel_chunks')
    synth_chunks_dir = os.path.join(path_gendir, 'synth_chunks')
    
    os.makedirs(chunks_dir, exist_ok=True)
    os.makedirs(mel_dir, exist_ok=True)
    os.makedirs(synth_chunks_dir, exist_ok=True)
    os.makedirs(os.path.join(path_gendir, 'pred'), exist_ok=True)
    
    if is_part:
        os.makedirs(os.path.join(path_gendir, 'part'), exist_ok=True)
    
    # Set parameters exactly as in dataset.py
    context_window_samples = int(args.data.duration * args.data.sampling_rate)
    hop_length = args.data.block_size
    win_length = 1024  # Same as in dataset.py
    n_mels = 80  # Same as in dataset.py
    fmin = 40  # Same as in dataset.py
    fmax = 12000  # Same as in dataset.py
    
    # Create mel transform with exactly the same parameters as in dataset.py
    device = torch.device(args.device)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=args.data.sampling_rate,
        n_fft=win_length,
        win_length=win_length,
        hop_length=hop_length,
        f_min=fmin,
        f_max=fmax,
        n_mels=n_mels,
        power=2.0
    ).to(device)
    
    def normalize_mel(mel_spec):
        """Same normalization as in dataset.py"""
        mel_spec = mel_spec - mel_spec.min()
        mel_spec = mel_spec / (mel_spec.max() + 1e-8)
        return mel_spec
    
    # Process in fixed-length chunks without overlap
    total_samples = len(audio)
    all_synthesized = []
    if is_part:
        all_harmonic = []
        all_noise = []
    
    # Initialize phase for continuity between chunks
    initial_phase = None
    
    for i in range(0, total_samples, context_window_samples):
        chunk_idx = i // context_window_samples + 1
        
        # Extract exact chunk
        if i + context_window_samples <= total_samples:
            # Full chunk
            chunk = audio[i:i+context_window_samples]
        else:
            # Last chunk might be smaller, so pad it
            chunk = np.zeros(context_window_samples)
            remaining = total_samples - i
            chunk[:remaining] = audio[i:total_samples]
        
        # Save input chunk
        chunk_path = os.path.join(chunks_dir, f"{fn}_chunk_{chunk_idx}.wav")
        sf.write(chunk_path, chunk, args.data.sampling_rate)
        print(f" > Processing chunk {chunk_idx}, samples {i}-{i+len(chunk)}")
        
        # Convert to tensor
        chunk_tensor = torch.FloatTensor(chunk).unsqueeze(0).to(device)
        
        # Extract mel spectrogram exactly as in dataset.py
        mel_spec = mel_transform(chunk_tensor)
        mel_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
        mel_norm = normalize_mel(mel_db)
        mel_np = mel_norm.squeeze(0).cpu().numpy().T  # Transpose to get (time, n_mels)
        
        # Save mel spectrogram
        mel_path = os.path.join(mel_dir, f"{fn}_chunk_{chunk_idx}.npy")
        np.save(mel_path, mel_np)
        
        # Convert mel back to tensor for inference
        mel_tensor = torch.from_numpy(mel_np).float().to(device).unsqueeze(0)
        
        # Forward pass with phase continuity
        with torch.no_grad():
            signal, f0_pred, final_phase, (s_h, s_n) = model(mel_tensor, initial_phase)
            # Update initial_phase for next chunk
            initial_phase = final_phase
        
        # Convert to numpy
        synth_chunk = utils.convert_tensor_to_numpy(signal)
        
        # Save synthesized chunk
        synth_chunk_path = os.path.join(synth_chunks_dir, f"{fn}_chunk_{chunk_idx}.wav")
        sf.write(synth_chunk_path, synth_chunk, args.data.sampling_rate)
        
        # For the final output, only use what corresponds to real input
        if i + context_window_samples > total_samples:
            valid_len = total_samples - i
            all_synthesized.append(synth_chunk[:valid_len])
            if is_part:
                harmonic_chunk = utils.convert_tensor_to_numpy(s_h)
                noise_chunk = utils.convert_tensor_to_numpy(s_n)
                all_harmonic.append(harmonic_chunk[:valid_len])
                all_noise.append(noise_chunk[:valid_len])
        else:
            all_synthesized.append(synth_chunk)
            if is_part:
                all_harmonic.append(utils.convert_tensor_to_numpy(s_h))
                all_noise.append(utils.convert_tensor_to_numpy(s_n))
    
    # Concatenate all chunks
    final_synthesized = np.concatenate(all_synthesized)
    
    # Save final output
    path_pred = os.path.join(path_gendir, 'pred', fn + '_synthesized.wav')
    print(f" > Saving final concatenated output to: {path_pred}")
    sf.write(path_pred, final_synthesized, args.data.sampling_rate)
    
    if is_part:
        path_pred_h = os.path.join(path_gendir, 'part', fn + '-harmonic.wav')
        path_pred_n = os.path.join(path_gendir, 'part', fn + '-noise.wav')
        
        final_harmonic = np.concatenate(all_harmonic)
        final_noise = np.concatenate(all_noise)
        
        sf.write(path_pred_h, final_harmonic, args.data.sampling_rate)
        sf.write(path_pred_n, final_noise, args.data.sampling_rate)
    
    print(" [*] Phase-continuous chunked inference complete.")


def render(args, model, path_mel_dir, path_gendir='gen', is_part=False):
    print(' [*] rendering...')
    model.eval()

    # list files
    files = utils.traverse_dir(
        path_mel_dir, 
        extension='npy', 
        is_ext=False,
        is_sort=True, 
        is_pure=True)
    num_files = len(files)
    print(' > num_files:', num_files)

    # run
    with torch.no_grad():
        for fidx in range(num_files):
            fn = files[fidx]
            print('--------')
            print('{}/{} - {}'.format(fidx, num_files, fn))

            path_mel = os.path.join(path_mel_dir, fn) + '.npy'
            mel = np.load(path_mel)
            mel = torch.from_numpy(mel).float().to(args.device).unsqueeze(0)
            print(' mel:', mel.shape)

            # forward
            signal, f0_pred, _, (s_h, s_n) = model(mel)

            # path
            path_pred = os.path.join(path_gendir, 'pred', fn + '.wav')
            if is_part:
                path_pred_n = os.path.join(path_gendir, 'part', fn + '-noise.wav')
                path_pred_h = os.path.join(path_gendir, 'part', fn + '-harmonic.wav')
            print(' > path_pred:', path_pred)
            
            os.makedirs(os.path.dirname(path_pred), exist_ok=True)
            if is_part:
                os.makedirs(os.path.dirname(path_pred_h), exist_ok=True)

            # to numpy
            pred = utils.convert_tensor_to_numpy(signal)
            if is_part:
                pred_n = utils.convert_tensor_to_numpy(s_n)
                pred_h = utils.convert_tensor_to_numpy(s_h)
            
            # save
            sf.write(path_pred, pred, args.data.sampling_rate)
            if is_part:
                sf.write(path_pred_n, pred_n, args.data.sampling_rate)
                sf.write(path_pred_h, pred_h, args.data.sampling_rate)


def test(args, model, loss_func, loader_test, path_gendir='gen', is_part=False):
    print(' [*] testing...')
    print(' [*] output folder:', path_gendir)
    model.eval()

    # losses
    test_loss = 0.
    test_loss_mss = 0.
    test_loss_f0 = 0.
    
    # intialization
    num_batches = len(loader_test)
    os.makedirs(path_gendir, exist_ok=True)
    rtf_all = []

    # run
    with torch.no_grad():
        for bidx, data in enumerate(loader_test):
            fn = data['name'][0]
            # Fix: Handle the case where fn is still a list
            if isinstance(fn, list):
                fn = fn[0]
                
            print('--------')
            print('{}/{} - {}'.format(bidx, num_batches, fn))

            # unpack data
            for k in data.keys():
                if k != 'name':
                    data[k] = data[k].to(args.device).float()
            print('>>', data['name'][0])

            # forward
            st_time = time.time()
            signal, f0_pred, _, (s_h, s_n) = model(data['mel'])
            ed_time = time.time()

            # crop
            min_len = np.min([signal.shape[1], data['audio'].shape[1]])
            signal        = signal[:,:min_len]
            data['audio'] = data['audio'][:,:min_len]

            # RTF
            run_time = ed_time - st_time
            song_time = data['audio'].shape[-1] / args.data.sampling_rate
            rtf = run_time / song_time
            print('RTF: {}  | {} / {}'.format(rtf, run_time, song_time))
            rtf_all.append(rtf)
           
            # loss
            loss, (loss_mss, loss_f0) = loss_func(
                signal, data['audio'], f0_pred, data['f0'])

            test_loss         += loss.item()
            test_loss_mss     += loss_mss.item() 
            test_loss_f0      += loss_f0.item()

            # path
            path_pred = os.path.join(path_gendir, 'pred', fn + '.wav')
            path_anno = os.path.join(path_gendir, 'anno', fn + '.wav')
            if is_part:
                path_pred_n = os.path.join(path_gendir, 'part', fn + '-noise.wav')
                path_pred_h = os.path.join(path_gendir, 'part', fn + '-harmonic.wav')

            print(' > path_pred:', path_pred)
            print(' > path_anno:', path_anno)

            os.makedirs(os.path.dirname(path_pred), exist_ok=True)
            os.makedirs(os.path.dirname(path_anno), exist_ok=True)
            if is_part:
                os.makedirs(os.path.dirname(path_pred_h), exist_ok=True)

            # to numpy
            pred  = utils.convert_tensor_to_numpy(signal)
            anno  = utils.convert_tensor_to_numpy(data['audio'])
            if is_part:
                pred_n = utils.convert_tensor_to_numpy(s_n)
                pred_h = utils.convert_tensor_to_numpy(s_h)
            
            # save
            sf.write(path_pred, pred, args.data.sampling_rate)
            sf.write(path_anno, anno, args.data.sampling_rate)
            if is_part:
                sf.write(path_pred_n, pred_n, args.data.sampling_rate)
                sf.write(path_pred_h, pred_h, args.data.sampling_rate)
            
    # report
    test_loss /= num_batches
    test_loss_mss     /= num_batches
    test_loss_f0      /= num_batches

    # check
    print(' [test_loss] test_loss:', test_loss)
    print(' Real Time Factor', np.mean(rtf_all))
    return test_loss, test_loss_mss, test_loss_f0


def train(args, model, loss_func, loader_train, loader_test, initial_global_step=-1):
    # saver
    saver = Saver(args, initial_global_step=initial_global_step)

    # model size
    params_count = utils.get_network_paras_amount({'model': model})
    saver.log_info('--- model size ---')
    saver.log_info(params_count)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.train.lr)

    # run
    best_loss = np.inf
    num_batches = len(loader_train)
    model.train()
    
    # Check for existing validation results to determine best_loss
    if os.path.exists(saver.path_log_value) and initial_global_step > 0:
        try:
            with open(saver.path_log_value, 'r') as f:
                for line in f:
                    if 'valid loss' in line:
                        parts = line.strip().split(' | ')
                        if len(parts) >= 2:
                            loss_val = float(parts[1])
                            if loss_val < best_loss:
                                best_loss = loss_val
            saver.log_info(f'Found previous best validation loss: {best_loss}')
        except Exception as e:
            saver.log_info(f'Error reading previous logs: {e}')
    
    saver.log_info('======= start training =======')
    saver.log_info(f'Starting from global step: {saver.global_step}')
    
    # Calculate starting epoch
    start_epoch = initial_global_step // num_batches if initial_global_step > 0 else 0
    saver.log_info(f'Starting from epoch: {start_epoch}')
    
    for epoch in range(start_epoch, args.train.epochs):
        saver.log_info(f'Beginning epoch {epoch}/{args.train.epochs}')
        epoch_loss = 0.0
        
        for batch_idx, data in enumerate(loader_train):
            saver.global_step_increment()
            optimizer.zero_grad()

            # unpack data
            for k in data.keys():
                if k != 'name':
                    data[k] = data[k].to(args.device).float()
            
            # forward
            signal, f0_pred, _, _,  = model(data['mel'])

            # loss
            loss, (loss_mss, loss_f0) = loss_func(
                signal, data['audio'], f0_pred, data['f0'])
            
            # handle nan loss
            if torch.isnan(loss):
                raise ValueError(' [x] nan loss ')
            else:
                # backpropagate
                loss.backward()
                optimizer.step()
                
            epoch_loss += loss.item()

            # log loss - still do this by steps for more granular monitoring
            if saver.global_step % args.train.interval_log == 0:
                saver.log_info(
                    'epoch: {}/{} {:3d}/{:3d} | {} | t: {:.2f} | loss: {:.6f} | time: {} | counter: {}'.format(
                        epoch,
                        args.train.epochs,
                        batch_idx,
                        num_batches,
                        saver.expdir,
                        saver.get_interval_time(),
                        loss.item(),
                        saver.get_total_time(),
                        saver.global_step
                    )
                )
                saver.log_info(
                    ' > mss loss: {:.6f}, f0: {:.6f}'.format(
                       loss_mss.item(),
                       loss_f0.item(),
                    )
                )

                y, s = signal, data['audio']
                saver.log_info(
                    "pred: max:{:.5f}, min:{:.5f}, mean:{:.5f}, rms: {:.5f}\n" \
                    "anno: max:{:.5f}, min:{:.5f}, mean:{:.5f}, rms: {:.5f}".format(
                            torch.max(y), torch.min(y), torch.mean(y), torch.mean(y** 2) ** 0.5,
                            torch.max(s), torch.min(s), torch.mean(s), torch.mean(s** 2) ** 0.5))

                saver.log_value({
                    'train loss': loss.item(), 
                    'train loss mss': loss_mss.item(),
                    'train loss f0': loss_f0.item(),
                })
        
        # End of epoch processing
        avg_epoch_loss = epoch_loss / num_batches
        saver.log_info(f'Epoch {epoch} completed with average loss: {avg_epoch_loss:.6f}')
        
        # Save model at specified epoch intervals
        if (epoch + 1) % args.train.interval_save == 0:
            saver.save_models(
                {'vocoder': model}, postfix=f'epoch_{epoch+1}')
            saver.log_info(f'Model saved at epoch {epoch+1}')
        
        # Validate at specified epoch intervals
        if (epoch + 1) % args.train.interval_val == 0:
            # run testing set
            path_testdir_runtime = os.path.join(
                    args.env.expdir,
                    'runtime_gen', 
                    f'gen_epoch_{epoch+1}')
            test_loss, test_loss_mss, test_loss_f0 = test(
                args, model, loss_func, loader_test,
                path_gendir=path_testdir_runtime)
            saver.log_info(
                ' --- <validation> --- \nloss: {:.6f}. mss loss: {:.6f}, f0: {:.6f}'.format(
                    test_loss, test_loss_mss, test_loss_f0
                )
            )

            saver.log_value({
                'valid loss': test_loss,
                'valid loss mss': test_loss_mss,
                'valid loss f0': test_loss_f0,
                'epoch': epoch + 1
            })
            model.train()

            # save best model
            if test_loss < best_loss:
                saver.log_info(f' [V] best model updated. Previous: {best_loss}, New: {test_loss}')
                saver.save_models(
                    {'vocoder': model}, postfix='best')
                best_loss = test_loss

            saver.make_report()