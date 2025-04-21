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
from solver import train, test, render

from ddsp.vocoder import SawSub, SawSinSub, Sins, DWS, Full
from ddsp.loss import HybridLoss


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
        help="Stages. Options: training/inference",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="Models. Options: SawSinSub/Sins/DWS/Full/SinsSub/SawSub",
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
    
    # Initialize batch tensors with correct dimensions
    batch_dict = {
        'name': [item['name'] for item in batch],
        'audio': torch.zeros(batch_size, max_audio_len),
        'mel': torch.stack([item['mel'] for item in batch]),
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
        
        batch_dict['audio'][i, :audio_len] = item['audio']
        
        # Handle f0 assignment based on its dimensions
        if len(f0_shape) == 1:  # f0 is 1D
            batch_dict['f0'][i, :f0_len] = item['f0']
        else:  # f0 is multi-dimensional
            batch_dict['f0'][i, :f0_len, ...] = item['f0']
    
    return batch_dict

# Define the DatasetWrapper at module level (outside any function)
class DatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        # The solver.py expects 'name' instead of 'filename'
        filename = item.pop('filename')
        # Handle the case where filename might already be a list
        if isinstance(filename, list):
            item['name'] = filename
        else:
            item['name'] = [filename]
        return item

def get_data_loaders(args, whole_audio=False):
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
        max_files=10000
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
    
    # Create data loaders with custom collate function
    train_loader = torch.utils.data.DataLoader(
        DatasetWrapper(train_dataset),
        batch_size=args.train.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=custom_collate
    )
    
    valid_loader = torch.utils.data.DataLoader(
        DatasetWrapper(valid_dataset),
        batch_size=args.inference.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=custom_collate
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

    else:
        raise ValueError(f" [x] Unknown Model: {cmd.model}")
    
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

    # loss
    loss_func = HybridLoss(args.loss.n_ffts)

    # device
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda':
        torch.cuda.set_device(args.env.gpu)
    model.to(args.device)
    loss_func.to(args.device)

    # datas
    loader_train, loader_valid = get_data_loaders(args, whole_audio=False)
        
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
    else:
          raise ValueError(f" [x] Unkown Stage: {cmd.stage }")