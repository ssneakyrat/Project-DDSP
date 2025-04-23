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
    """Simplified collate function that assumes all tensors are pre-padded."""
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
            # For tensor data - simple stacking, no padding needed
            batch_dict[key] = torch.stack([sample[key] for sample in batch])
    
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
    
    # Train model
    trainer.fit(model, train_loader, val_loader)
    
    print("Training completed successfully.")

if __name__ == "__main__":
    main()