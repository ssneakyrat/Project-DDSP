import os
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import torchaudio
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim import AdamW

from model.synth import Synth
from model.loss import HybridLoss

class LightningModel(pl.LightningModule):
    def __init__(self, config, phone_map_len, singer_map_len, language_map_len):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Get memory optimization settings
        self.use_gradient_checkpointing = config['model'].get('use_gradient_checkpointing', True)
        self.use_mixed_precision = config['training'].get('precision', '32') in ['16', '16-mixed']
        
        # Initialize enhanced synthesizer model with the new inputs
        self.model = Synth(
            sampling_rate=config['model']['sample_rate'],
            block_size=config['model']['hop_length'],
            n_mag_harmonic=config['model']['n_mag_harmonic'],
            n_mag_noise=config['model']['n_mag_noise'],
            n_harmonics=config['model']['n_harmonics'],
            phone_map_len=phone_map_len,
            singer_map_len=singer_map_len,
            language_map_len=language_map_len,
            n_formants=config['model'].get('n_formants', 4),
            n_breath_bands=config['model'].get('n_breath_bands', 8),
            n_mels=config['model'].get('n_mels', 80),
            use_gradient_checkpointing=self.use_gradient_checkpointing
        )

        # Initialize loss function with mel spectrogram loss
        self.loss_fn = HybridLoss(
            use_mel_loss=config['loss'].get('use_mel_loss', False),
            use_mss_loss=config['loss'].get('use_mss_loss', True),
            use_f0_loss=config['loss'].get('use_f0_loss', True),
            use_amplitude_loss=config['loss'].get('use_amplitude_loss', False),
            amplitude_weight=config['loss'].get('amplitude_weight', 0.1),
            n_ffts=config['loss']['n_ffts'],
            sample_rate=config['model']['sample_rate'],
            n_mels=config['model']['n_mels'],
            mel_weight=config['loss'].get('mel_weight', 0.5),
        )
        
        # Initialize automatic mixed precision scaler if needed
        if self.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
            
        # Initialize mel loss tracking for guidance ratio
        self.current_mel_loss = None
        
        # Get mel guidance settings from config
        self.mel_guidance_enabled = config.get('mel_guidance', {}).get('enabled', False)
        self.mel_guidance_upper = config.get('mel_guidance', {}).get('upper_threshold', 2.0)
        self.mel_guidance_lower = config.get('mel_guidance', {}).get('lower_threshold', 0.5)
        self.mel_guidance_warmup = config.get('mel_guidance', {}).get('warmup_epochs', 10)
        self.mel_guidance_scheduler = config.get('mel_guidance', {}).get('scheduler', 'linear')
    
    def forward(self, batch):
        # Determine guidance ratio based on current mel loss
        guidance_ratio = self._get_guidance_ratio()
        
        # Forward with calculated guidance ratio
        return self.model(batch, guidance_ratio=guidance_ratio)
    
    def _get_guidance_ratio(self):
        """Calculate the current guidance ratio based on mel loss and training progress"""
        # Skip guidance if disabled
        if not self.mel_guidance_enabled:
            return 0.0
            
        # Always use full guidance during warmup period
        if self.current_epoch < self.mel_guidance_warmup:
            return 1.0
            
        # If no mel loss recorded yet, default to full guidance
        if self.current_mel_loss is None:
            return 1.0
            
        # Calculate guidance ratio based on current mel loss
        upper = self.mel_guidance_upper
        lower = self.mel_guidance_lower
        
        if self.current_mel_loss >= upper:
            # Loss too high, use full guidance
            guidance_ratio = 1.0
        elif self.current_mel_loss <= lower:
            # Loss low enough, use no guidance
            guidance_ratio = 0.0
        else:
            # Linear interpolation between thresholds
            guidance_ratio = (self.current_mel_loss - lower) / (upper - lower)
            
            # Apply non-linear scheduling if configured
            if self.mel_guidance_scheduler == 'exponential':
                # Exponential decay - faster reduction in guidance
                guidance_ratio = guidance_ratio ** 2
                
        # Clamp to valid range
        guidance_ratio = min(1.0, max(0.0, guidance_ratio))
        
        return guidance_ratio
    
    def training_step(self, batch, batch_idx):
        # Get current guidance ratio
        guidance_ratio = self._get_guidance_ratio()
        
        # Use mixed precision where available for training
        if self.use_mixed_precision and not self.trainer.precision.startswith("bf16"):
            with torch.cuda.amp.autocast():
                # Forward with guidance ratio
                signal, f0_pred, _, components = self.model(batch, guidance_ratio=guidance_ratio)
                
                # Extract pre-computed mel if available
                signal_mel = components[3] if len(components) > 3 else None
                
                # Extract harmonic amplitudes if amplitude loss is enabled
                amplitudes_pred = None
                if self.loss_fn.use_amplitude_loss and len(components) > 2:
                    _, _, amplitudes_pred = components[:3]
                
                # Compute Loss with pre-computed mel
                loss_dict = self.loss_fn(
                    signal, batch['audio'], 
                    f0_pred, batch['f0'], 
                    mel_input=batch.get('mel', None),
                    signal_mel=signal_mel,
                    amplitudes_pred=amplitudes_pred,
                    amplitudes_true=batch.get('amplitudes', None)
                )
                
                # Extract total loss
                total_loss = loss_dict['loss']
        else:
            # Standard precision forward with guidance ratio
            signal, f0_pred, _, components = self.model(batch, guidance_ratio=guidance_ratio)
            
            # Extract pre-computed mel if available
            signal_mel = components[3] if len(components) > 3 else None
            
            # Extract harmonic amplitudes if amplitude loss is enabled
            amplitudes_pred = None
            if self.loss_fn.use_amplitude_loss and len(components) > 2:
                _, _, amplitudes_pred = components[:3]
            
            # Compute Loss with pre-computed mel
            loss_dict = self.loss_fn(
                signal, batch['audio'], 
                f0_pred, batch['f0'], 
                mel_input=batch.get('mel', None),
                signal_mel=signal_mel,
                amplitudes_pred=amplitudes_pred,
                amplitudes_true=batch.get('amplitudes', None)
            )
            
            # Extract total loss
            total_loss = loss_dict['loss']

        # Store current mel loss for next iteration
        if 'loss_mel' in loss_dict:
            self.current_mel_loss = loss_dict['loss_mel'].item()

        # Log losses
        for loss_name, loss_value in loss_dict.items():
            self.log(f'train_{loss_name}', loss_value, prog_bar=True, batch_size=self.config['dataset']['batch_size'])
        
        # Log guidance ratio
        self.log('guidance_ratio', guidance_ratio, prog_bar=True, batch_size=self.config['dataset']['batch_size'])
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        # For validation, always use a low guidance ratio to test true model performance
        # Use 0.2 instead of 0 to provide minimal guidance during validation
        guidance_ratio = 0.2
        
        # Standard precision forward
        signal, f0_pred, _, components = self.model(batch, guidance_ratio=guidance_ratio)
        
        # Extract pre-computed mel if available
        signal_mel = components[3] if len(components) > 3 else None
        
        # Extract harmonic amplitudes if amplitude loss is enabled
        amplitudes_pred = None
        if self.loss_fn.use_amplitude_loss and len(components) > 2:
            _, _, amplitudes_pred = components[:3]
        
        # Compute Loss with pre-computed mel
        loss_dict = self.loss_fn(
            signal, batch['audio'], 
            f0_pred, batch['f0'], 
            mel_input=batch.get('mel', None),
            signal_mel=signal_mel,
            amplitudes_pred=amplitudes_pred,
            amplitudes_true=batch.get('amplitudes', None)
        )
        
        # Extract total loss
        total_loss = loss_dict['loss']
        
        # Log losses
        for loss_name, loss_value in loss_dict.items():
            self.log(f'val_{loss_name}', loss_value, prog_bar=True, batch_size=self.config['dataset']['batch_size'])
        
        # Log audio for first few batches
        if batch_idx < 3:  # Limit to save space
            self._log_audio(batch, signal, 'val')

        return total_loss
    
    def configure_optimizers(self):
        # Create optimizer groups with different learning rates
        core_params = []
        expression_params = []
        mel_refiner_params = []
        
        # Separate parameters for potentially different learning rates
        for name, param in self.named_parameters():
            if 'expression_predictor' in name:
                expression_params.append(param)
            elif 'mel_refiner' in name:
                mel_refiner_params.append(param)
            else:
                core_params.append(param)
        
        # Get learning rates from config
        base_lr = self.config['training']['learning_rate']
        expression_lr_factor = self.config['training'].get('expression_lr_factor', 1.5)
        mel_refiner_lr_factor = self.config['training'].get('mel_refiner_lr_factor', 2.0)
        
        # Create optimizer with parameter groups
        optimizer = AdamW([
            {'params': core_params, 'lr': base_lr},
            {'params': expression_params, 'lr': base_lr * expression_lr_factor},
            {'params': mel_refiner_params, 'lr': base_lr * mel_refiner_lr_factor}
        ], weight_decay=self.config['training']['weight_decay'])
        
        # Create learning rate scheduler
        scheduler = ExponentialLR(
            optimizer,
            gamma=self.config['training']['lr_scheduler']['gamma']
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }