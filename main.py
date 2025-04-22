'''
author: Hsiao Wen Yi (wayne391)
email:  s101062219@gmail.com
'''

import os
import argparse
import torch
import shutil
import time
import glob
from logger import utils, report
from solver import train, test, render, is_svs_model
from ddsp.vocoder import SawSub, SawSinSub, Sins, DWS, Full
from ddsp.loss import HybridLoss
from ddsp.pseudo_mel import PseudoMelGenerator, FormantParameterPredictor
from ddsp.svs_vocoder import SVSVocoder
from ddsp.svs_loss import SVSHybridLoss

def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="path to config file",
    )
    parser.add_argument(
        "-s",
        "--stage",
        type=str,
        required=True,
        help="Stages. Options: training/inference/validation/wav_inference",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="Models. Options: SawSinSub/Sins/DWS/Full/SinsSub/SawSub/SVSFormant",
    )
    parser.add_argument(
        "-k",
        "--model_ckpt",
        type=str,
        required=False,
        help="path to existing model ckpt",
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        required=False,
        help="[inference] path to input mel-spectrogram",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=False,
        help="[inference, validation] path to synthesized audio files",
    )
    parser.add_argument(
        "-p",
        "--is_part",
        type=str,
        required=False,
        help="[inference, validation] individual harmonic and noise output",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start a fresh training run (discard previous checkpoints)",
    )
    parser.add_argument(
        "-w",
        "--wav_file",
        type=str,
        required=False,
        help="[wav_inference] path to input WAV file",
    )
    return parser.parse_args(args=args, namespace=namespace)

def custom_collate(batch):
    """
    Custom collate function to handle variable-length tensors in a batch.
    """
    # Get batch size
    batch_size = len(batch)
    
    # Extract dimensions from first item to understand tensor shapes
    example_item = batch[0]
    f0_shape = example_item['f0'].shape
    mel_shape = example_item['mel'].shape
    
    # Find max lengths in the batch
    max_audio_len = max(item['audio'].size(0) for item in batch)
    max_f0_len = max(item['f0'].size(0) for item in batch)
    max_mel_len = max(item['mel'].size(0) for item in batch)
    
    # Initialize batch tensors with correct dimensions
    batch_dict = {
        'name': [item['name'] for item in batch],
        'audio': torch.zeros(batch_size, max_audio_len),
        'mel': torch.zeros(batch_size, max_mel_len, mel_shape[1]),
    }
    
    # Handle f0 based on its actual dimensions
    if len(f0_shape) == 1:  # f0 is 1D
        batch_dict['f0'] = torch.zeros(batch_size, max_f0_len)
    else:  # f0 is multi-dimensional
        batch_dict['f0'] = torch.zeros(batch_size, max_f0_len, *f0_shape[1:])
    
    # Fill tensors with actual data (with padding)
    for i, item in enumerate(batch):
        audio_len = item['audio'].size(0)
        f0_len = item['f0'].size(0)
        mel_len = item['mel'].size(0)
        
        batch_dict['audio'][i, :audio_len] = item['audio']
        batch_dict['mel'][i, :mel_len] = item['mel']
        
        # Handle f0 assignment based on its dimensions
        if len(f0_shape) == 1:  # f0 is 1D
            batch_dict['f0'][i, :f0_len] = item['f0']
        else:  # f0 is multi-dimensional
            batch_dict['f0'][i, :f0_len, ...] = item['f0']
    
    return batch_dict

def svs_collate_fn(batch):
    """
    Custom collate function for SVS dataset that handles variable-length tensors.
    Keeps phoneme and mel frames aligned without unnecessary padding/truncation.
    """
    # Get batch size
    batch_size = len(batch)
    
    # Initialize output dictionary with lists
    collated = {
        'name': [item['name'] for item in batch],
    }
    
    # Find max lengths for various sequence fields
    max_audio_len = max(item['audio'].size(0) for item in batch)
    max_phoneme_len = max(item['phonemes'].size(0) for item in batch)
    max_f0_len = max(item['f0'].size(0) for item in batch)
    max_duration_len = max(item['durations'].size(0) for item in batch)
    
    # IMPORTANT: Make sure max_f0_len matches the expected mel frames length
    # This ensures all conditioning parameters are aligned to mel frames
    
    # Initialize tensors with proper shapes and types
    collated['audio'] = torch.zeros(batch_size, max_audio_len)
    collated['phonemes'] = torch.zeros(batch_size, max_phoneme_len, dtype=torch.long)
    collated['f0'] = torch.zeros(batch_size, max_f0_len)
    collated['durations'] = torch.zeros(batch_size, max_duration_len)
    
    # Process singer_id and language_id - ensure these are long tensors
    collated['singer_id'] = torch.zeros(batch_size, dtype=torch.long)
    collated['language_id'] = torch.zeros(batch_size, dtype=torch.long)
    
    # If mel spectrograms are included
    if 'mel' in batch[0]:
        mel_dim = batch[0]['mel'].size(-1)
        max_mel_len = max(item['mel'].size(0) for item in batch)
        collated['mel'] = torch.zeros(batch_size, max_mel_len, mel_dim)
        
        # Ensure max_f0_len matches max_mel_len for proper alignment
        if max_f0_len != max_mel_len:
            print(f"Warning: max_f0_len ({max_f0_len}) does not match max_mel_len ({max_mel_len})")
            # Recreate f0 tensor with proper length
            collated['f0'] = torch.zeros(batch_size, max_mel_len)
    
    # Fill the tensors with actual data
    for i, item in enumerate(batch):
        # Handle variable-length sequences with proper padding
        audio_len = item['audio'].size(0)
        phoneme_len = item['phonemes'].size(0)
        f0_len = item['f0'].size(0)
        duration_len = item['durations'].size(0)
        
        collated['audio'][i, :audio_len] = item['audio']
        collated['phonemes'][i, :phoneme_len] = item['phonemes']
        
        # Ensure f0 is padded to match mel length if mel is present
        if 'mel' in item:
            mel_len = item['mel'].size(0)
            # Use the smaller of f0_len and mel_len to avoid index errors
            valid_f0_len = min(f0_len, mel_len)
            max_target_len = max_mel_len
            
            collated['f0'][i, :valid_f0_len] = item['f0'][:valid_f0_len]
            collated['mel'][i, :mel_len] = item['mel']
        else:
            collated['f0'][i, :f0_len] = item['f0']
            
        collated['durations'][i, :duration_len] = item['durations']
        
        # Handle scalar values - ensure correct types
        collated['singer_id'][i] = item['singer_id'].long() if isinstance(item['singer_id'], torch.Tensor) else torch.tensor(item['singer_id'], dtype=torch.long)
        collated['language_id'][i] = item['language_id'].long() if isinstance(item['language_id'], torch.Tensor) else torch.tensor(item['language_id'], dtype=torch.long)
    
    return collated

# Define the DatasetWrapper at module level (outside any function)
class DatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        # The solver.py expects 'name' instead of 'filename'
        filename = item.pop('filename') if 'filename' in item else f"sample_{idx}"
        # Handle the case where filename might already be a list
        if isinstance(filename, list):
            item['name'] = filename
        else:
            item['name'] = [filename]
        return item
    
import numpy as np

class SVSDatasetWrapper(torch.utils.data.Dataset):
    """
    Dataset wrapper for Singing Voice Synthesis
    
    Takes the existing dataset and transforms it to the format required by the SVS vocoder
    Ensures phoneme sequences and durations are correctly aligned with mel frames
    """
    def __init__(self, dataset):
        """
        Initialize the SVS dataset wrapper
        
        Args:
            dataset: The base dataset containing audio, phonemes, F0, singer IDs, etc.
        """
        self.dataset = dataset
        self.hop_length = 240  # Same as in dataset.py
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Extract and ensure consistent format for all required fields
        filename = item.get('filename', f"sample_{idx}")
        
        # Get singer_id and ensure it's a tensor of the correct type (long)
        if 'singer_id' in item:
            singer_id = item['singer_id']
            if isinstance(singer_id, (list, np.ndarray)):
                singer_id = singer_id[0] if len(singer_id) > 0 else 0
            singer_id = torch.tensor(singer_id, dtype=torch.long)
        else:
            singer_id = torch.tensor(0, dtype=torch.long)
        
        # Get language_id and ensure it's a tensor of the correct type (long)
        if 'language_id' in item:
            language_id = item['language_id']
            if isinstance(language_id, (list, np.ndarray)):
                language_id = language_id[0] if len(language_id) > 0 else 0
            language_id = torch.tensor(language_id, dtype=torch.long)
        else:
            language_id = torch.tensor(0, dtype=torch.long)
        
        # Extract phoneme sequence - now this is already in mel frames in dataset.py
        if 'phone_seq' in item:
            phone_seq = item['phone_seq']
            # Ensure phone_seq is a tensor of the correct type
            if not isinstance(phone_seq, torch.Tensor):
                phone_seq = torch.tensor(phone_seq, dtype=torch.long)
            elif phone_seq.dtype != torch.long:
                phone_seq = phone_seq.long()
        else:
            # If no phone_seq, create dummy one matching mel length
            if 'mel' in item:
                mel_len = item['mel'].shape[0]
                phone_seq = torch.zeros(mel_len, dtype=torch.long)
            else:
                phone_seq = torch.zeros(100, dtype=torch.long)
        
        # Get mel data
        mel = item['mel']
        if not isinstance(mel, torch.Tensor):
            mel = torch.tensor(mel, dtype=torch.float)
        
        # Get F0 data and ensure it's a float tensor aligned with mel frames
        if 'f0' in item:
            f0 = item['f0']
            if not isinstance(f0, torch.Tensor):
                f0 = torch.tensor(f0, dtype=torch.float)
            
            # Ensure f0 length matches mel frames
            if len(f0) != len(mel):
                # Resize f0 to match mel length
                if len(f0) > len(mel):
                    f0 = f0[:len(mel)]
                else:
                    # Pad f0 if shorter
                    pad_f0 = torch.zeros(len(mel), dtype=torch.float)
                    pad_f0[:len(f0)] = f0
                    f0 = pad_f0
        else:
            # Create dummy F0 matching mel length
            f0 = torch.ones(len(mel), dtype=torch.float) * 220.0
        
        # Calculate durations by counting consecutive same phonemes
        # This approach assumes phone_seq is already in frame-level
        if phone_seq.dim() == 1:
            # Count duration of each unique phoneme
            durations = []
            current_phone = None
            current_count = 0
            
            for phone in phone_seq:
                phone_item = phone.item()
                if current_phone is None:
                    current_phone = phone_item
                    current_count = 1
                elif phone_item == current_phone:
                    current_count += 1
                else:
                    durations.append(current_count)
                    current_phone = phone_item
                    current_count = 1
            
            # Add the last phoneme duration
            if current_count > 0:
                durations.append(current_count)
            
            # Convert to tensor
            if durations:
                durations = torch.tensor(durations, dtype=torch.float)
            else:
                # If no durations found, create dummy
                durations = torch.ones(1, dtype=torch.float)
            
            # Create phoneme sequence with one entry per unique phoneme
            unique_phones = []
            current_phone = None
            
            for phone in phone_seq:
                phone_item = phone.item()
                if current_phone is None or phone_item != current_phone:
                    current_phone = phone_item
                    unique_phones.append(phone_item)
            
            phonemes = torch.tensor(unique_phones, dtype=torch.long)
        else:
            # Fallback for unexpected phone_seq format
            phonemes = torch.zeros(1, dtype=torch.long)
            durations = torch.ones(1, dtype=torch.float) * len(mel)
        
        # Ensure audio is a float tensor
        audio = item['audio']
        if not isinstance(audio, torch.Tensor):
            audio = torch.tensor(audio, dtype=torch.float)
        
        # Construct the output with all required fields for SVS
        return {
            'name': filename,
            'audio': audio,
            'phonemes': phonemes,  # Now contains unique phonemes
            'durations': durations,  # Frame-level durations for each unique phoneme
            'f0': f0,  # Frame-level F0
            'singer_id': singer_id,
            'language_id': language_id,
            'mel': mel  # Frame-level mel spectrogram
        }

def get_data_loaders(args, is_svs_model=False, whole_audio=False):
    """
    Get data loaders for training and validation.
    """
    import torch
    from dataset import SingingVoiceDataset
    
    # Create training dataset
    train_dataset = SingingVoiceDataset(
        dataset_dir=args.data.train_path,
        cache_dir="./cache/train",
        sample_rate=args.data.sampling_rate,
        context_window_samples=int(args.data.duration * args.data.sampling_rate),
        rebuild_cache=False,
        n_mels=80,
        hop_length=args.data.block_size,
        win_length=1024,
        max_files=None
    )
    
    # Create validation dataset
    valid_dataset = SingingVoiceDataset(
        dataset_dir=args.data.valid_path,
        cache_dir="./cache/valid",
        sample_rate=args.data.sampling_rate,
        context_window_samples=int(args.data.duration * args.data.sampling_rate),
        rebuild_cache=False,
        n_mels=80,
        hop_length=args.data.block_size,
        win_length=1024,
        max_files=10
    )
    
    # Choose the appropriate wrapper and collate function based on the model type
    if is_svs_model:
        wrapper_cls = SVSDatasetWrapper
        collate_fn = svs_collate_fn
        print(" > Using SVS dataset wrapper and collate function")
    else:
        wrapper_cls = DatasetWrapper
        collate_fn = custom_collate
        print(" > Using standard dataset wrapper and collate function")
    
    # Create data loaders with appropriate wrapper and collate function
    train_loader = torch.utils.data.DataLoader(
        wrapper_cls(train_dataset),
        batch_size=args.train.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn
    )
    
    valid_loader = torch.utils.data.DataLoader(
        wrapper_cls(valid_dataset),
        batch_size=args.inference.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn
    )
    
    return train_loader, valid_loader

def find_latest_checkpoint(expdir):
    """Find the latest checkpoint in the experiment directory."""
    ckpt_dir = os.path.join(expdir, 'ckpts')
    if not os.path.exists(ckpt_dir):
        return None
    
    # Look for checkpoint files with a step number in the name
    ckpt_files = glob.glob(os.path.join(ckpt_dir, 'vocoder_*.pt'))
    if not ckpt_files:
        return None
    
    # Extract step numbers and find the latest
    latest_ckpt = None
    latest_step = -1
    
    for ckpt_file in ckpt_files:
        # Try to extract step number from filename
        try:
            filename = os.path.basename(ckpt_file)
            # Format typically like: vocoder_12345_2.pt or vocoder_best.pt
            if 'best' in filename:
                continue  # Skip best model as it's not a specific step
            step_str = filename.split('_')[1]
            step = int(step_str)
            if step > latest_step:
                latest_step = step
                latest_ckpt = ckpt_file
        except (IndexError, ValueError):
            continue
    
    return latest_ckpt

def handle_experiment_directory(args, fresh_start=False):
    """Handle experiment directory creation or loading."""
    expdir = args.env.expdir
    
    if os.path.exists(expdir):
        if fresh_start:
            print(f"Starting fresh: Removing existing experiment directory: {expdir}")
            shutil.rmtree(expdir)
            os.makedirs(expdir, exist_ok=True)
            return None
        else:
            print(f"Experiment directory exists: {expdir}")
            # Find latest checkpoint
            latest_ckpt = find_latest_checkpoint(expdir)
            if latest_ckpt:
                print(f"Found latest checkpoint: {latest_ckpt}")
            else:
                print("No checkpoints found, starting from scratch.")
            return latest_ckpt
    else:
        print(f"Creating new experiment directory: {expdir}")
        os.makedirs(expdir, exist_ok=True)
        return None

if __name__ == '__main__':
    # parse commands
    cmd = parse_args()
    
    # load config
    args = utils.load_config(cmd.config)
    print(' > config:', cmd.config)
    print(' >    exp:', args.env.expdir)
    
    # Handle experiment directory and find checkpoint to resume from
    latest_ckpt = None
    if cmd.stage == 'training':
        latest_ckpt = handle_experiment_directory(args, fresh_start=cmd.fresh)
        # If user specified a checkpoint, use that instead of the latest found
        if cmd.model_ckpt:
            latest_ckpt = cmd.model_ckpt
    
    # load model
    model = None
    if cmd.model == 'SawSinSub':
        model = SawSinSub(
            sampling_rate=args.data.sampling_rate,
            block_size=args.data.block_size,
            n_mag_harmonic=args.model.n_mag_harmonic,
            n_mag_noise=args.model.n_mag_noise,
            n_harmonics=args.model.n_harmonics)

    elif cmd.model == 'Sins':
        model = Sins(
            sampling_rate=args.data.sampling_rate,
            block_size=args.data.block_size,
            n_harmonics=args.model.n_harmonics,
            n_mag_noise=args.model.n_mag_noise)

    elif cmd.model == 'DWS':
        model = DWS(
            sampling_rate=args.data.sampling_rate,
            block_size=args.data.block_size,
            num_wavetables=args.model.num_wavetables,
            len_wavetables=args.model.len_wavetables,
            is_lpf=args.model.is_lpf)

    elif cmd.model == 'Full':
        model = Full(
            sampling_rate=args.data.sampling_rate,
            block_size=args.data.block_size,
            n_mag_harmonic=args.model.n_mag_harmonic,
            n_mag_noise=args.model.n_mag_noise,
            n_harmonics=args.model.n_harmonics)

    elif cmd.model == 'SawSub':
        model = SawSub(
            sampling_rate=args.data.sampling_rate,
            block_size=args.data.block_size)
    elif cmd.model == 'SVSFormant':
        # Get dataset to determine vocabulary sizes if not specified
        from dataset import SingingVoiceDataset
        temp_dataset = SingingVoiceDataset(
            dataset_dir=args.data.train_path,
            cache_dir="./cache/temp",
            rebuild_cache=False,
        )
        
        # Use dataset info or config values
        num_phonemes = getattr(args.model, 'num_phonemes', len(temp_dataset.phone_map) + 1)  # +1 for padding
        num_singers = getattr(args.model, 'num_singers', len(temp_dataset.singer_map))
        num_languages = getattr(args.model, 'num_languages', len(temp_dataset.language_map))
        
        print(f" > Using vocabulary sizes: {num_phonemes} phones, {num_singers} singers, {num_languages} languages")
        
        model = SVSVocoder(
            sampling_rate=args.data.sampling_rate,
            block_size=args.data.block_size,
            n_mag_harmonic=args.model.n_mag_harmonic,
            n_mag_noise=args.model.n_mag_noise,
            n_harmonics=args.model.n_harmonics,
            num_phonemes=num_phonemes,
            num_singers=num_singers,
            num_languages=num_languages,
            n_mels=getattr(args.model, 'pseudo_mel_dim', 80),
            n_formants=getattr(args.model, 'n_formants', 5)
        )
        
        # Use the SVS hybrid loss instead of normal hybrid loss
        loss_func = SVSHybridLoss(args.loss.n_ffts, args.data.sampling_rate)
    else:
        raise ValueError(f" [x] Unknown Model: {cmd.model}")
    
    # Check if we're using an SVS model
    using_svs_model = is_svs_model(model)
    
    # Set default loss function if not already set
    if 'loss_func' not in locals():
        loss_func = HybridLoss(args.loss.n_ffts)
    
    # load parameters from checkpoint if available
    initial_global_step = -1
    if latest_ckpt:
        print(f"Loading model from checkpoint: {latest_ckpt}")
        model = utils.load_model_params(latest_ckpt, model, args.device)
        # Try to extract step number from checkpoint filename for continuing training
        try:
            ckpt_basename = os.path.basename(latest_ckpt)
            if '_' in ckpt_basename:
                step_str = ckpt_basename.split('_')[1]
                initial_global_step = int(step_str)
                print(f"Continuing training from step {initial_global_step}")
        except (IndexError, ValueError):
            initial_global_step = -1
    # For inference or validation modes, always use the specified checkpoint
    elif cmd.model_ckpt:
        print(f"Loading model from specified checkpoint: {cmd.model_ckpt}")
        model = utils.load_model_params(cmd.model_ckpt, model, args.device)

    # device
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda':
        torch.cuda.set_device(args.env.gpu)
    model.to(args.device)
    loss_func.to(args.device)

    # Only load datasets for stages that need them
    loader_train = None
    loader_valid = None
    if cmd.stage in ['training', 'validation']:
        # Load data loaders with appropriate mode for the model type
        print(" > Loading datasets for", cmd.stage)
        loader_train, loader_valid = get_data_loaders(args, is_svs_model=using_svs_model, whole_audio=False)
        
    # stage
    if cmd.stage == 'training':
        train(args, model, loss_func, loader_train, loader_valid, initial_global_step=initial_global_step)
    elif cmd.stage == 'validation':
        output_dir = 'valid_gen'
        if cmd.output_dir:
            output_dir = cmd.output_dir
        test(
            args, 
            model, 
            loss_func, 
            loader_valid, 
            path_gendir=output_dir,
            is_part=cmd.is_part)
    elif cmd.stage == 'inference':
        output_dir = 'infer_gen'
        if cmd.output_dir:
            output_dir = cmd.output_dir
        render(
            args, 
            model, 
            path_mel_dir=cmd.input_dir, 
            path_gendir=output_dir,
            is_part=cmd.is_part)
    elif cmd.stage == 'wav_inference':
        if not cmd.model_ckpt:
            raise ValueError(" [x] --model_ckpt is required for wav_inference stage")
        if not cmd.wav_file:
            raise ValueError(" [x] --wav_file is required for wav_inference stage")
        
        output_dir = 'wav_infer_gen'
        if cmd.output_dir:
            output_dir = cmd.output_dir
        
        # Import the new function here to ensure it's available
        from solver import inference_from_wav
        
        inference_from_wav(
            args,
            model,
            path_wav_file=cmd.wav_file,
            path_gendir=output_dir,
            is_part=cmd.is_part)
    else:
          raise ValueError(f" [x] Unknown Stage: {cmd.stage }")