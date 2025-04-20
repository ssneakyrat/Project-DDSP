'''
author: Hsiao Wen Yi (wayne391)
email:  s101062219@gmail.com
'''

import os
import argparse
import torch

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
    return parser.parse_args(args=args, namespace=namespace)

# Define the DatasetWrapper at module level (outside any function)
class DatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        # The solver.py expects 'name' instead of 'filename'
        # Also wrapping in a list since solver.py accesses it as data['name'][0]
        item['name'] = [item.pop('filename')]
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
        max_files=1000
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
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        DatasetWrapper(train_dataset),
        batch_size=args.train.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    valid_loader = torch.utils.data.DataLoader(
        DatasetWrapper(valid_dataset),
        batch_size=args.inference.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    
    return train_loader, valid_loader

if __name__ == '__main__':
    # parse commands
    cmd = parse_args()
    
    # load config
    args = utils.load_config(cmd.config)
    print(' > config:', cmd.config)
    print(' >    exp:', args.env.expdir)

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
    
    # load parameters
    if cmd.model_ckpt:
        model = utils.load_model_params(
            cmd.model_ckpt, model, args.device)

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
        train(args, model, loss_func, loader_train, loader_valid)
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
    