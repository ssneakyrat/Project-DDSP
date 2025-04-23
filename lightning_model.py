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
        
        # Initialize enhanced synthesizer model
        self.model = Synth(
            sampling_rate=config['model']['sample_rate'],
            block_size=config['model']['hop_length'],
            n_mag_harmonic=config['model']['n_mag_harmonic'],
            n_mag_noise=config['model']['n_mag_noise'],
            n_harmonics=config['model']['n_harmonics'],
            n_formants=config['model'].get('n_formants', 4),
            n_breath_bands=config['model'].get('n_breath_bands', 8),
            n_mels=config['model'].get('n_mels', 80),
            use_gradient_checkpointing=self.use_gradient_checkpointing
        )

        # Initialize loss function with mel spectrogram loss
        self.loss_fn = HybridLoss(
            use_mel_loss=config['loss']['use_mel_loss'],
            n_ffts=config['loss']['n_ffts'],
            sample_rate=config['model']['sample_rate'],
            n_mels=config['model']['n_mels'],
            mel_weight=config['loss']['mel_weight'],
        )
        
        # Initialize automatic mixed precision scaler if needed
        if self.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
    
    def forward(self, batch):
        return self.model(batch['mel'])
    
    def training_step(self, batch, batch_idx):
        # Use mixed precision where available for training
        if self.use_mixed_precision and not self.trainer.precision.startswith("bf16"):
            with torch.cuda.amp.autocast():
                # Forward
                signal, f0_pred, _, _ = self(batch)
                
                # Compute Loss
                loss_dict = self.loss_fn(
                    signal, batch['audio'], f0_pred, batch['f0'], mel_input=batch['mel'])
                
                # Extract total loss
                total_loss = loss_dict['loss']
        else:
            # Standard precision forward
            signal, f0_pred, _, _ = self(batch)
            
            # Compute Loss
            loss_dict = self.loss_fn(
                signal, batch['audio'], f0_pred, batch['f0'], mel_input=batch['mel'])
            
            # Extract total loss
            total_loss = loss_dict['loss']

        # Log losses
        for loss_name, loss_value in loss_dict.items():
            self.log(f'train_{loss_name}', loss_value, prog_bar=True, batch_size=self.config['dataset']['batch_size'])
        
        # Log audio occasionally
        if self.current_epoch % self.config['logging'].get('audio_log_every_n_epoch', 10) == 0 and batch_idx % 50 == 0:
            self._log_audio(batch, signal, 'train')

        return total_loss
    
    def validation_step(self, batch, batch_idx):
        # Forward
        signal, f0_pred, _, _ = self(batch)

        # Compute Loss
        loss_dict = self.loss_fn(
            signal, batch['audio'], f0_pred, batch['f0'], mel_input=batch['mel'])
        
        # Log losses
        for loss_name, loss_value in loss_dict.items():
            self.log(f'val_{loss_name}', loss_value, prog_bar=True, batch_size=self.config['dataset']['batch_size'])
        
        # Extract total loss
        total_loss = loss_dict['loss']
        
        # Log audio for first few batches
        if batch_idx < 3:  # Limit to save space
            self._log_audio(batch, signal, 'val')

        return total_loss
    
    def _log_audio(self, batch, signal, stage):
        if not hasattr(self.logger, 'experiment'):
            return
        
        # Get sample rate from config
        sample_rate = self.config['model']['sample_rate']
        
        # Only log a limited number of samples
        num_samples = min(2, batch['audio'].size(0))  # Reduced from 3 to 2 to save space
        
        for idx in range(num_samples):
            # Original audio
            audio_orig = batch['audio'][idx].detach().cpu().numpy()
            
            # Generated audio from model
            audio_gen = signal[idx].detach().cpu().numpy()
            
            # Ensure audio is in the correct range for TensorBoard (-1 to 1)
            max_orig = max(abs(audio_orig.min()), abs(audio_orig.max()))
            max_gen = max(abs(audio_gen.min()), abs(audio_gen.max()))
            
            if max_orig > 0:
                audio_orig = audio_orig / max_orig
            if max_gen > 0:
                audio_gen = audio_gen / max_gen
            
            # Log original audio
            self.logger.experiment.add_audio(
                f'{stage}_original_audio_{idx}',
                audio_orig,
                self.global_step,
                sample_rate=sample_rate
            )
            
            # Log generated audio
            self.logger.experiment.add_audio(
                f'{stage}_generated_audio_{idx}',
                audio_gen,
                self.global_step,
                sample_rate=sample_rate
            )
        
        # Create and log audio waveform visualization
        fig = self._create_audio_waveform_figure(batch, signal, num_samples)
        self.logger.experiment.add_figure(f'{stage}_audio_comparison', fig, self.global_step)
        plt.close(fig)

    def _create_audio_waveform_figure(self, batch, signal, num_samples):
        """Create a figure comparing original and generated audio waveforms."""
        # Create a more compact visualization
        fig, axes = plt.subplots(num_samples, 2, figsize=(10, 3 * num_samples))
        
        if num_samples == 1:
            axes = [axes]  # Make it iterable for the loop
            
        for idx in range(num_samples):
            # Original audio
            audio_orig = batch['audio'][idx].detach().cpu().numpy()
            # Generated audio
            audio_gen = signal[idx].detach().cpu().numpy()
            
            # Plot original waveform (use subsampling for efficiency)
            subsample = max(1, len(audio_orig) // 1000)  # Limit plot points
            time_orig = np.arange(0, len(audio_orig), subsample) / self.config['model']['sample_rate']
            axes[idx][0].plot(time_orig, audio_orig[::subsample])
            axes[idx][0].set_title('Original Audio')
            axes[idx][0].set_xlabel('Time (s)')
            
            # Plot generated waveform
            time_gen = np.arange(0, len(audio_gen), subsample) / self.config['model']['sample_rate']
            axes[idx][1].plot(time_gen, audio_gen[::subsample])
            axes[idx][1].set_title('Generated Audio')
            axes[idx][1].set_xlabel('Time (s)')
        
        plt.tight_layout()
        return fig
    
    def configure_optimizers(self):
        # Create optimizer groups with different learning rates
        core_params = []
        expression_params = []
        
        # Separate parameters for potentially different learning rates
        for name, param in self.named_parameters():
            if 'expression_predictor' in name:
                expression_params.append(param)
            else:
                core_params.append(param)
        
        # Get learning rates from config
        base_lr = self.config['training']['learning_rate']
        expression_lr_factor = self.config['training'].get('expression_lr_factor', 1.5)
        
        # Create optimizer with parameter groups
        optimizer = AdamW([
            {'params': core_params, 'lr': base_lr},
            {'params': expression_params, 'lr': base_lr * expression_lr_factor}
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
    
    def on_save_checkpoint(self, checkpoint):
        """Custom saving logic to reduce checkpoint size"""
        # Save a smaller version of the model for inference
        if hasattr(self, 'model') and getattr(self, 'global_step', 0) % 1000 == 0:
            save_path = os.path.join(
                self.config['logging']['checkpoint_dir'], 
                self.config['logging']['name'],
                f'model_step_{self.global_step}.pt'
            )
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save only the model weights without optimizer state
            torch.save(self.model.state_dict(), save_path)
            
        return checkpoint