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

from model.dummy_model import DummyModel
from model.dummy_loss import DummyLoss

class LightningModel(pl.LightningModule):
    def __init__(self, config, phone_map_len, singer_map_len, language_map_len):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Initialize model
        self.model = Synth(
            sampling_rate=config['model']['sample_rate'],
            block_size=config['model']['hop_length'],
            n_mag_harmonic=config['model']['n_mag_harmonic'],
            n_mag_noise=config['model']['n_mag_noise'],
            n_harmonics=config['model']['n_harmonics']
        )

        self.loss_fn = HybridLoss(config['loss']['n_ffts'])

        ''' dummy
        # Initialize model
        self.model = DummyModel(
            hidden_size=config['model']['hidden_size'],
            n_layers=config['model']['n_layers'],
            dropout=config['model']['dropout'],
            n_mels=config['model']['n_mels'],
            phone_map_len=phone_map_len,
            singer_map_len=singer_map_len,
            language_map_len=language_map_len
        )
        
        # Initialize loss function
        self.loss_fn = DummyLoss()
        
        # Track metrics
        self.train_loss = 0.0
        self.val_loss = 0.0
        '''
        
    def forward(self, batch):
        return self.model(batch['mel'])
        '''
        return self.model(
            mel=batch['mel'],
            phone_seq=batch['phone_seq'],
            f0=batch['f0'],
            singer_id=batch['singer_id'],
            language_id=batch['language_id']
        )
        '''
    
    def training_step(self, batch, batch_idx):

        # Forward
        signal, f0_pred, _, _,  = self(batch)
        
        #print('signal:', signal.shape)
        #print('f0_pred:', f0_pred.shape)

        loss_dict = self.loss_fn(
            signal, batch['audio'], f0_pred, batch['f0'])

        # Log losses
        for loss_name, loss_value in loss_dict.items():
            self.log(f'train_{loss_name}', loss_value, prog_bar=True, batch_size=self.config['dataset']['batch_size'])
        
        # Log total loss
        total_loss = loss_dict['loss']
        self.train_loss = total_loss
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):

        # Forward
        signal, f0_pred, _, _,  =self(batch)

        # Compute Loss
        loss_dict = self.loss_fn(
            signal, batch['audio'], f0_pred, batch['f0'])
        
        # Log losses
        for loss_name, loss_value in loss_dict.items():
            self.log(f'val_{loss_name}', loss_value, prog_bar=True, batch_size=self.config['dataset']['batch_size'])
        
        # Log total loss
        total_loss = loss_dict['loss']
        self.val_loss = total_loss
        
        self._log_audio(batch, signal, 'val')

        return total_loss
    
    def _log_audio(self, batch, outputs, stage):
        if not hasattr(self.logger, 'experiment'):
            return
        
        # Get sample rate from config
        sample_rate = self.config['model']['sample_rate']
        
        # Extract signals from model outputs (signal, f0_pred, _, _)
        signal = outputs[0] if isinstance(outputs, tuple) else outputs
        
        # Only log a few samples to avoid cluttering TensorBoard
        num_samples = min(3, batch['audio'].size(0))
        
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
        if num_samples == 1:
            # When num_samples=1, create a 1x2 grid
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Original audio
            audio_orig = batch['audio'][0].detach().cpu().numpy()
            # Generated audio
            audio_gen = signal[0].detach().cpu().numpy()
            
            # Plot original waveform
            time_orig = np.arange(len(audio_orig)) / self.config['model']['sample_rate']
            ax1.plot(time_orig, audio_orig)
            ax1.set_title('Original Audio Waveform')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Amplitude')
            
            # Plot generated waveform
            time_gen = np.arange(len(audio_gen)) / self.config['model']['sample_rate']
            ax2.plot(time_gen, audio_gen)
            ax2.set_title('Generated Audio Waveform')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Amplitude')
        else:
            # When num_samples>1, create an nx2 grid
            fig, axes = plt.subplots(num_samples, 2, figsize=(12, 4 * num_samples))
            
            for idx in range(num_samples):
                # Original audio
                audio_orig = batch['audio'][idx].detach().cpu().numpy()
                # Generated audio
                audio_gen = signal[idx].detach().cpu().numpy()
                
                # Plot original waveform
                time_orig = np.arange(len(audio_orig)) / self.config['model']['sample_rate']
                axes[idx, 0].plot(time_orig, audio_orig)
                axes[idx, 0].set_title(f'Original Audio Waveform {idx}')
                axes[idx, 0].set_xlabel('Time (s)')
                axes[idx, 0].set_ylabel('Amplitude')
                
                # Plot generated waveform
                time_gen = np.arange(len(audio_gen)) / self.config['model']['sample_rate']
                axes[idx, 1].plot(time_gen, audio_gen)
                axes[idx, 1].set_title(f'Generated Audio Waveform {idx}')
                axes[idx, 1].set_xlabel('Time (s)')
                axes[idx, 1].set_ylabel('Amplitude')
        
        plt.tight_layout()
        return fig
    def _log_samples(self, batch, outputs, stage):
        if not hasattr(self.logger, 'experiment'):
            return
        
        # Only log first sample in batch
        idx = 0
        
        # Log original and reconstructed mel spectrograms
        fig, ax = plt.subplots(2, 1, figsize=(10, 6))
        
        # Original mel
        mel_orig = batch['mel'][idx].detach().cpu().numpy()
        ax[0].imshow(mel_orig, origin='lower', aspect='auto')
        ax[0].set_title('Original Mel Spectrogram')
        ax[0].set_ylabel('Mel bins')
        
        # Reconstructed mel
        mel_recon = outputs['mel_output'][idx].float().detach().cpu().numpy()
        ax[1].imshow(mel_recon, origin='lower', aspect='auto')
        ax[1].set_title('Reconstructed Mel Spectrogram')
        ax[1].set_xlabel('Frames')
        ax[1].set_ylabel('Mel bins')
        
        # Add to tensorboard
        self.logger.experiment.add_figure(f'{stage}_mel_comparison', fig, self.global_step)
        plt.close(fig)
    
    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
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