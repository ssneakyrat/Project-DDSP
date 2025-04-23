import os
import argparse
import yaml
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

from dataset import get_dataloader, SingingVoiceDataset
from lightning_model import LightningModel

def simplified_collate(batch):
    """Robust collate function that handles tensors of different sizes."""
    # Filter out any None values
    batch = [x for x in batch if x is not None]
    if not batch:
        return None
    
    # Create dictionary for batched data
    batch_dict = {}
    
    # Handle each key
    for key in batch[0].keys():
        if key == 'filename':
            # For non-tensor data
            batch_dict[key] = [sample[key] for sample in batch]
        else:
            # Check if all tensors have the same shape
            shapes = [sample[key].shape for sample in batch]
            if len(set(str(s) for s in shapes)) > 1:
                #print(f"WARNING: Inconsistent shapes for {key}: {shapes}")
                
                # For handling different dimensions
                if key in ['audio', 'phone_seq', 'f0_audio']:
                    # Audio-aligned tensors should be 1D with length 48000
                    target_size = 48000
                    for i, sample in enumerate(batch):
                        if sample[key].shape[0] != target_size:
                            #print(f"  Resizing {key} tensor {i} from {sample[key].shape[0]} to {target_size}")
                            if sample[key].shape[0] < target_size:
                                # Pad
                                batch[i][key] = torch.nn.functional.pad(sample[key], (0, target_size - sample[key].shape[0]))
                            else:
                                # Truncate
                                batch[i][key] = sample[key][:target_size]
                
                elif key in ['phone_seq_mel', 'f0']:
                    # Mel-aligned 1D tensors
                    target_size = 201  # 48000 // 240 + 1 (for 2-second audio with hop_length=240)
                    for i, sample in enumerate(batch):
                        if sample[key].shape[0] != target_size:
                            #print(f"  Resizing {key} tensor {i} from {sample[key].shape[0]} to {target_size}")
                            if sample[key].shape[0] < target_size:
                                # Pad
                                batch[i][key] = torch.nn.functional.pad(sample[key], (0, target_size - sample[key].shape[0]))
                            else:
                                # Truncate
                                batch[i][key] = sample[key][:target_size]
                
                elif key == 'mel':
                    # Mel is 2D [frames, n_mels]
                    target_frames = 201  # Same as mel-aligned 1D tensors
                    n_mels = batch[0][key].shape[1]  # Keep n_mels dimension the same
                    
                    for i, sample in enumerate(batch):
                        if sample[key].shape[0] != target_frames:
                            #print(f"  Resizing {key} tensor {i} from {sample[key].shape} to [{target_frames}, {n_mels}]")
                            if sample[key].shape[0] < target_frames:
                                # Pad
                                batch[i][key] = torch.nn.functional.pad(sample[key], (0, 0, 0, target_frames - sample[key].shape[0]))
                            else:
                                # Truncate
                                batch[i][key] = sample[key][:target_frames, :]
                
                elif key == 'amplitudes':
                    # Amplitudes is 2D [frames, n_harmonics]
                    target_frames = 201  # Same as mel-aligned tensors
                    n_harmonics = batch[0][key].shape[1]  # Keep n_harmonics dimension the same
                    
                    for i, sample in enumerate(batch):
                        if sample[key].shape[0] != target_frames:
                            #print(f"  Resizing {key} tensor {i} from {sample[key].shape} to [{target_frames}, {n_harmonics}]")
                            if sample[key].shape[0] < target_frames:
                                # Pad
                                batch[i][key] = torch.nn.functional.pad(sample[key], (0, 0, 0, target_frames - sample[key].shape[0]))
                            else:
                                # Truncate
                                batch[i][key] = sample[key][:target_frames, :]
                
                elif key in ['phone_one_hot', 'phone_mel_one_hot']:
                    # One-hot encoded tensors
                    # Handle according to their specific dimensions
                    # For phone_one_hot: [audio_length, n_phones]
                    # For phone_mel_one_hot: [mel_frames, n_phones]
                    
                    if key == 'phone_one_hot':
                        target_length = 48000
                    else:  # phone_mel_one_hot
                        target_length = 201
                    
                    n_classes = batch[0][key].shape[1]
                    
                    for i, sample in enumerate(batch):
                        if sample[key].shape[0] != target_length:
                            #print(f"  Resizing {key} tensor {i} from {sample[key].shape} to [{target_length}, {n_classes}]")
                            if sample[key].shape[0] < target_length:
                                # Pad
                                batch[i][key] = torch.nn.functional.pad(sample[key], (0, 0, 0, target_length - sample[key].shape[0]))
                            else:
                                # Truncate
                                batch[i][key] = sample[key][:target_length, :]
                
                # For other tensors like singer_id, language_id, etc.
                # These should already be consistent, but just in case:
                '''
                else:
                    most_common_shape = max(shapes, key=shapes.count)
                    for i, sample in enumerate(batch):
                        if sample[key].shape != most_common_shape:
                            #print(f"  Unexpected shape mismatch for {key}: {sample[key].shape} vs {most_common_shape}")
                            # You could implement reshaping here if needed
                '''
            
            try:
                # Stack the tensors after ensuring consistent shapes
                batch_dict[key] = torch.stack([sample[key] for sample in batch])
            except RuntimeError as e:
                print(f"ERROR stacking {key} tensors. Shapes: {[sample[key].shape for sample in batch]}")
                raise e
    
    return batch_dict

def parse_args():
    parser = argparse.ArgumentParser(description='Train singing voice model')
    parser.add_argument('--config', type=str, default='configs/model.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint for resuming')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--val_ratio', type=float, default=0.02, help='Validation data ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for train/val split')
    return parser.parse_args()

def main():
    #torch.set_float32_matmul_precision('medium')

    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Modify config for debug mode
    if args.debug:
        config['dataset']['max_files'] = 10
        config['training']['epochs'] = 3
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("Loading full dataset...")
    
    # Load the full dataset once
    full_dataset = SingingVoiceDataset(
        rebuild_cache=False, 
        max_files=config['dataset']['max_files'] if 'max_files' in config['dataset'] else None,
        n_mels=config['dataset'].get('n_mels', 80),
        hop_length=config['dataset'].get('hop_length', 240),
        win_length=config['dataset'].get('win_length', 1024),
        fmin=config['dataset'].get('fmin', 40),
        fmax=config['dataset'].get('fmax', 12000),
        num_workers=config['dataset']['num_workers'],
        device='cuda' if torch.cuda.is_available() else 'cpu',
        n_harmonics=config['model']['n_harmonics'],
    )
    
    # Calculate train/val split sizes
    full_dataset_size = len(full_dataset)
    val_size = int(full_dataset_size * args.val_ratio)
    train_size = full_dataset_size - val_size
    
    print(f"Full dataset size: {full_dataset_size}")
    print(f"Train set size: {train_size}")
    print(f"Validation set size: {val_size}")
    
    # Create indices for train/val split
    indices = list(range(full_dataset_size))
    # Shuffle indices using the set seed
    indices = torch.randperm(full_dataset_size).tolist()
    
    # Split indices for train and validation
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    # Create samplers for train and validation
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    
    # Create data loaders with the samplers
    train_loader = torch.utils.data.DataLoader(
        full_dataset,
        batch_size=config['dataset']['batch_size'],
        sampler=train_sampler,
        num_workers=config['dataset']['num_workers'],
        pin_memory=config['dataset']['pin_memory'],
        persistent_workers=config['dataset']['persistent_workers'] if config['dataset']['num_workers'] > 0 else False,
        collate_fn=simplified_collate
    )
    
    val_loader = torch.utils.data.DataLoader(
        full_dataset,  # Use the same dataset
        batch_size=config['dataset']['batch_size'],
        sampler=val_sampler,
        num_workers=config['dataset']['num_workers'],
        pin_memory=config['dataset']['pin_memory'],
        persistent_workers=config['dataset']['persistent_workers'] if config['dataset']['num_workers'] > 0 else False,
        collate_fn=simplified_collate
    )
    
    # Calculate len of phone map for model configuration
    phone_map_len = len(full_dataset.phone_map) + 1  # +1 for padding
    singer_map_len = len(full_dataset.singer_map)
    language_map_len = len(full_dataset.language_map)
    
    print(f"Phone map length: {phone_map_len}")
    print(f"Singer map length: {singer_map_len}")
    print(f"Language map length: {language_map_len}")
    
    # Create model
    model = LightningModel(
        config=config,
        phone_map_len=phone_map_len,
        singer_map_len=singer_map_len,
        language_map_len=language_map_len
    )
    
    # Set up logging
    logger = TensorBoardLogger(
        save_dir=config['logging']['save_dir'],
        name=config['logging']['name']
    )
    
    # Create callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(config['logging']['checkpoint_dir'], config['logging']['name']),
            filename='{epoch}-{val_loss:.4f}',
            save_top_k=config['logging']['save_top_k'],
            monitor=config['logging']['monitor'],
            mode=config['logging']['mode']
        ),
        LearningRateMonitor(logging_interval='step')
    ]
    
    # Add early stopping callback if configured
    if config['logging'].get('early_stopping', False):
        callbacks.append(
            EarlyStopping(
                monitor=config['logging']['monitor'],
                mode=config['logging']['mode'],
                patience=config['logging'].get('patience', 10),
                verbose=True
            )
        )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['epochs'],
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=config['training']['gradient_clip_val'],
        precision=config['training']['precision'],
        log_every_n_steps=config['logging']['log_every_n_steps'],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        check_val_every_n_epoch=config['logging']['check_val_every_n_epoch'],
        num_sanity_val_steps=0
    )
    
    # Train model - now with checkpoint support
    if args.checkpoint:
        print(f"Resuming training from checkpoint: {args.checkpoint}")
        trainer.fit(model, train_loader, val_loader, ckpt_path=args.checkpoint)
    else:
        trainer.fit(model, train_loader, val_loader)
    
    print("Training completed successfully.")

if __name__ == "__main__":
    main()