import os
import argparse
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

from dataset import get_dataloader
from lightning_model import LightningModel

def custom_collate(batch):
    # Filter out any None values
    batch = [x for x in batch if x is not None]
    if not batch:
        return None
        
    # Get elements and keys
    elem = batch[0]
    batch_dict = {key: [] for key in elem.keys()}
    
    # Group by key
    for sample in batch:
        for key, value in sample.items():
            batch_dict[key].append(value)
    
    # Special handling for tensors and strings
    output_batch = {}
    for key, values in batch_dict.items():
        if key == 'filename':
            # Handle non-tensor data
            output_batch[key] = values
        else:
            # Stack tensors with same shape, pad if needed
            try:
                output_batch[key] = torch.stack(values)
            except:
                # If stacking fails, pad to max length
                if key in ['audio', 'phone_seq', 'phone_one_hot', 'f0', 'mel']:
                    # Find max length in dimension 0
                    max_len = max([v.size(0) for v in values])
                    # Pad each tensor to max length
                    padded_values = []
                    for v in values:
                        if v.size(0) < max_len:
                            if v.dim() == 1:
                                padding = torch.zeros(max_len - v.size(0), dtype=v.dtype)
                            else:
                                padding_shape = (max_len - v.size(0),) + v.size()[1:]
                                padding = torch.zeros(padding_shape, dtype=v.dtype)
                            v = torch.cat([v, padding])
                        padded_values.append(v)
                    output_batch[key] = torch.stack(padded_values)
                else:
                    # For other tensor types, just pass the list
                    output_batch[key] = values
                    
    return output_batch

def parse_args():
    parser = argparse.ArgumentParser(description='Train singing voice model')
    parser.add_argument('--config', type=str, default='configs/model.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint for resuming')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    return parser.parse_args()

def main():

    torch.set_float32_matmul_precision('medium')

    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Modify config for debug mode
    if args.debug:
        config['dataset']['max_files'] = 10
        config['training']['epochs'] = 3
    
    # Create data loaders
    train_loader, dataset = get_dataloader(
        batch_size=config['dataset']['batch_size'],
        num_workers=config['dataset']['num_workers'],
        pin_memory=config['dataset']['pin_memory'],
        persistent_workers=config['dataset']['persistent_workers'],
        max_files=config['dataset']['train_files'],
        collate_fn=custom_collate
    )
    
    # Create data loaders
    val_loader, dataset_val = get_dataloader(
        batch_size=config['dataset']['batch_size'],
        num_workers=config['dataset']['num_workers'],
        pin_memory=config['dataset']['pin_memory'],
        persistent_workers=config['dataset']['persistent_workers'],
        max_files=config['dataset']['val_files'],
        collate_fn=custom_collate,
        shuffle=False
    )

    # Calculate len of phone map for model configuration
    phone_map_len = len(dataset.phone_map) + 1  # +1 for padding
    singer_map_len = len(dataset.singer_map)
    language_map_len = len(dataset.language_map)
    
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
    )
    
    # Train model
    trainer.fit(model, train_loader, val_loader)
    
    print("Training completed successfully.")

if __name__ == "__main__":
    main()