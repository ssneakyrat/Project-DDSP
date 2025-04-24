import os
import argparse
import yaml
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

from dataset import get_dataloader
from utils import standardized_collate_fn
from lightning_model import LightningModel

def parse_args():
    parser = argparse.ArgumentParser(description='Train singing voice model')
    parser.add_argument('--config', type=str, default='configs/model.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint for resuming')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Modify config for debug mode
    if args.debug:
        config['dataset']['train_files'] = 5
        config['dataset']['val_files'] = 2
        config['training']['epochs'] = 3
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("Creating separate train and validation datasets...")
    
    # Create separate train and validation dataloaders with standardized collate function
    train_loader, val_loader, train_dataset, val_dataset = get_dataloader(
        batch_size=config['dataset']['batch_size'],
        num_workers=config['dataset']['num_workers'],
        pin_memory=config['dataset']['pin_memory'],
        persistent_workers=config['dataset']['persistent_workers'],
        train_files=config['dataset'].get('train_files', None),
        val_files=config['dataset'].get('val_files', None),
        device='cuda' if torch.cuda.is_available() else 'cpu',
        collate_fn=standardized_collate_fn,  # Use the standardized collate function from utils.py
        n_harmonics=config['model']['n_harmonics'],
        context_window_sec=config['model']['context_window_sec'],  # Pass context window from config
        seed=args.seed
    )
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Max audio length: {train_dataset.max_audio_length} samples ({train_dataset.max_audio_length/train_dataset.sample_rate:.2f} seconds)")
    print(f"Max mel frames: {train_dataset.max_mel_frames}")
    
    # Calculate len of phone map for model configuration (use train dataset for this)
    phone_map_len = len(train_dataset.phone_map) + 1  # +1 for padding
    singer_map_len = len(train_dataset.singer_map)
    language_map_len = len(train_dataset.language_map)
    
    print(f"Phone map length: {phone_map_len}")
    print(f"Singer map length: {singer_map_len}")
    print(f"Language map length: {language_map_len}")
    
    # Verify that train and validation datasets have compatible maps
    if (len(train_dataset.phone_map) != len(val_dataset.phone_map) or 
        len(train_dataset.singer_map) != len(val_dataset.singer_map) or
        len(train_dataset.language_map) != len(val_dataset.language_map)):
        print("WARNING: Train and validation datasets have different map sizes!")
        print(f"Train phone map: {len(train_dataset.phone_map)}, Val phone map: {len(val_dataset.phone_map)}")
        print(f"Train singer map: {len(train_dataset.singer_map)}, Val singer map: {len(val_dataset.singer_map)}")
        print(f"Train language map: {len(train_dataset.language_map)}, Val language map: {len(val_dataset.language_map)}")
    
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