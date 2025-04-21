import os
import time
import torch
import numpy as np
import soundfile as sf

from logger.saver import Saver
from logger import utils
from ddsp.svs_vocoder import SVSVocoder
from ddsp.svs_loss import SVSHybridLoss
from svs_dataset import get_svs_data_loaders

def train_svs(args, model, loss_func, loader_train, loader_test, initial_global_step=-1):
    """
    Training function for SVS
    
    Args:
        args: Configuration arguments
        model: SVS vocoder model
        loss_func: Loss function
        loader_train: Training data loader
        loader_test: Validation data loader
        initial_global_step: Starting step for training
    """
    # Initialize saver
    saver = Saver(args, initial_global_step=initial_global_step)
    
    # Log model size
    params_count = utils.get_network_paras_amount({'svs_vocoder': model})
    saver.log_info('--- model size ---')
    saver.log_info(params_count)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.train.lr)
    
    # Training loop
    best_loss = np.inf
    num_batches = len(loader_train)
    model.train()
    
    for epoch in range(args.train.epochs):
        for batch_idx, data in enumerate(loader_train):
            saver.global_step_increment()
            optimizer.zero_grad()
            
            # Unpack data and move to device
            for k in data.keys():
                if k != 'name':
                    if k in ['phonemes', 'singer_id', 'language_id']:
                        data[k] = data[k].to(args.device).long()
                    else:
                        data[k] = data[k].to(args.device).float()
            
            # Forward pass
            signal, f0_pred, _, _, _ = model(
                data['phonemes'], 
                data['durations'], 
                data['f0'], 
                data['singer_id'],
                data['language_id']
            )
            
            # Compute loss
            loss, (loss_mss, loss_f0, loss_mel) = loss_func(
                signal, data['audio'], f0_pred, data['f0']
            )
            
            # Handle NaN loss
            if torch.isnan(loss):
                saver.log_info(f'NaN loss detected at step {saver.global_step}, skipping batch')
                continue
                
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            # Logging
            if saver.global_step % args.train.interval_log == 0:
                saver.log_info(
                    'epoch: {}/{} {:3d}/{:3d} | loss: {:.6f}'.format(
                        epoch, args.train.epochs, batch_idx, num_batches, loss.item()
                    )
                )
                saver.log_info(
                    ' > mss: {:.6f}, f0: {:.6f}, mel: {:.6f}'.format(
                        loss_mss.item(), loss_f0.item(), loss_mel.item()
                    )
                )
                saver.log_value({
                    'train loss': loss.item(),
                    'train loss mss': loss_mss.item(),
                    'train loss f0': loss_f0.item(),
                    'train loss mel': loss_mel.item(),
                })
                
        # Save model checkpoint
        if (epoch + 1) % args.train.interval_save == 0:
            saver.save_models({'svs_vocoder': model}, postfix=f'{saver.global_step}')
                
        # Validation
        if (epoch + 1) % args.train.interval_val == 0:
            val_loss = validate_svs(args, model, loss_func, loader_test, saver)
            
            # Save model if improved
            if val_loss < best_loss:
                saver.log_info(f' [V] best model updated: {val_loss}')
                saver.save_models({'svs_vocoder': model}, postfix='best')
                best_loss = val_loss
                
    saver.log_info(f'Training completed. Best validation loss: {best_loss}')
    
def validate_svs(args, model, loss_func, loader_test, saver=None):
    """
    Validation function for SVS
    
    Args:
        args: Configuration arguments
        model: SVS vocoder model
        loss_func: Loss function
        loader_test: Validation data loader
        saver: Optional saver for logging
        
    Returns:
        avg_loss: Average validation loss
    """
    model.eval()
    total_loss = 0.0
    total_loss_mss = 0.0
    total_loss_f0 = 0.0
    total_loss_mel = 0.0
    
    # Create validation output directory
    if saver:
        val_dir = os.path.join(args.env.expdir, 'validation', f'step_{saver.global_step}')
        os.makedirs(val_dir, exist_ok=True)
    else:
        val_dir = None
    
    with torch.no_grad():
        for batch_idx, data in enumerate(loader_test):
            # Unpack data and move to device
            for k in data.keys():
                if k != 'name':
                    data[k] = data[k].to(args.device).float()
            
            # Forward pass
            signal, f0_pred, _, components, _ = model(
                data['phonemes'], 
                data['durations'], 
                data['f0'], 
                data['singer_id'],
                data['language_id']
            )
            
            # Compute loss
            loss, (loss_mss, loss_f0, loss_mel) = loss_func(
                signal, data['audio'], f0_pred, data['f0']
            )
            
            # Accumulate losses
            total_loss += loss.item()
            total_loss_mss += loss_mss.item()
            total_loss_f0 += loss_f0.item()
            total_loss_mel += loss_mel.item()
            
            # Save audio samples
            if val_dir and batch_idx < 5:  # Save first 5 samples
                sample_name = data['name'][0] if isinstance(data['name'], list) else data['name']
                
                # Convert tensors to numpy
                audio_pred = utils.convert_tensor_to_numpy(signal)
                audio_true = utils.convert_tensor_to_numpy(data['audio'])
                harmonic, noise = components
                harmonic = utils.convert_tensor_to_numpy(harmonic)
                noise = utils.convert_tensor_to_numpy(noise)
                
                # Save files
                sf.write(os.path.join(val_dir, f'{sample_name}_pred.wav'), 
                         audio_pred, args.data.sampling_rate)
                sf.write(os.path.join(val_dir, f'{sample_name}_true.wav'), 
                         audio_true, args.data.sampling_rate)
                sf.write(os.path.join(val_dir, f'{sample_name}_harmonic.wav'), 
                         harmonic, args.data.sampling_rate)
                sf.write(os.path.join(val_dir, f'{sample_name}_noise.wav'), 
                         noise, args.data.sampling_rate)
    
    # Calculate averages
    num_batches = len(loader_test)
    avg_loss = total_loss / num_batches
    avg_loss_mss = total_loss_mss / num_batches
    avg_loss_f0 = total_loss_f0 / num_batches
    avg_loss_mel = total_loss_mel / num_batches
    
    # Log validation results
    if saver:
        saver.log_info(f'Validation loss: {avg_loss:.6f}')
        saver.log_info(f' > mss: {avg_loss_mss:.6f}, f0: {avg_loss_f0:.6f}, mel: {avg_loss_mel:.6f}')
        
        saver.log_value({
            'valid loss': avg_loss,
            'valid loss mss': avg_loss_mss,
            'valid loss f0': avg_loss_f0,
            'valid loss mel': avg_loss_mel,
        })
    
    model.train()
    return avg_loss

def inference_svs(args, model, input_dir, output_dir):
    """
    Inference function for SVS
    
    Args:
        args: Configuration arguments
        model: SVS vocoder model
        input_dir: Directory containing input phonemes, F0, etc.
        output_dir: Directory to save generated audio
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    # List input files
    files = utils.traverse_dir(input_dir, extension='npz', is_sort=True)
    
    with torch.no_grad():
        for file_path in files:
            # Load input data
            data = np.load(file_path)
            phonemes = torch.from_numpy(data['phonemes']).to(args.device).long().unsqueeze(0)
            durations = torch.from_numpy(data['durations']).to(args.device).float().unsqueeze(0)
            f0 = torch.from_numpy(data['f0']).to(args.device).float().unsqueeze(0)
            singer_id = torch.tensor([data['singer_id']]).to(args.device).long()
            language_id = torch.tensor([data['language_id']]).to(args.device).long()
            
            # Generate audio
            signal, _, _, components, _ = model(phonemes, durations, f0, singer_id, language_id)
            
            # Convert to numpy
            audio = utils.convert_tensor_to_numpy(signal)
            harmonic, noise = components
            harmonic = utils.convert_tensor_to_numpy(harmonic)
            noise = utils.convert_tensor_to_numpy(noise)
            
            # Save files
            base_name = os.path.basename(file_path).split('.')[0]
            sf.write(os.path.join(output_dir, f'{base_name}.wav'), 
                     audio, args.data.sampling_rate)
            sf.write(os.path.join(output_dir, f'{base_name}_harmonic.wav'), 
                     harmonic, args.data.sampling_rate)
            sf.write(os.path.join(output_dir, f'{base_name}_noise.wav'), 
                     noise, args.data.sampling_rate)
            
            print(f'Generated {base_name}.wav')

def main():
    """
    Main function for SVS training and inference
    """
    import argparse
    from dataset import SingingVoiceDataset
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--mode', type=str, choices=['train', 'inference'], required=True)
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint for inference')
    parser.add_argument('--input_dir', type=str, help='Path to input directory for inference')
    parser.add_argument('--output_dir', type=str, help='Path to output directory for inference')
    args = parser.parse_args()
    
    # Load config
    config = utils.load_config(args.config)
    
    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config.device = device
    
    # Create dataset and data loaders
    if args.mode == 'train':
        train_loader, valid_loader = get_svs_data_loaders(config, SingingVoiceDataset)
        
        # Create model and loss function
        model = SVSVocoder(
            sampling_rate=config.data.sampling_rate,
            block_size=config.data.block_size,
            n_mag_harmonic=config.model.n_mag_harmonic,
            n_mag_noise=config.model.n_mag_noise,
            n_harmonics=config.model.n_harmonics,
            num_phonemes=len(train_loader.dataset.dataset.phone_map) + 1,  # +1 for padding
            num_singers=len(train_loader.dataset.dataset.singer_map),
            num_languages=len(train_loader.dataset.dataset.language_map)
        ).to(device)
        
        loss_func = SVSHybridLoss(
            n_ffts=config.loss.n_ffts,
            sample_rate=config.data.sampling_rate
        ).to(device)
        
        # Train model
        train_svs(config, model, loss_func, train_loader, valid_loader)
        
    elif args.mode == 'inference':
        # Load dataset to get mappings
        dataset = SingingVoiceDataset(
            dataset_dir=config.data.train_path,
            cache_dir="./cache/temp",
            sample_rate=config.data.sampling_rate,
            rebuild_cache=False
        )
        
        # Create model
        model = SVSVocoder(
            sampling_rate=config.data.sampling_rate,
            block_size=config.data.block_size,
            n_mag_harmonic=config.model.n_mag_harmonic,
            n_mag_noise=config.model.n_mag_noise,
            n_harmonics=config.model.n_harmonics,
            num_phonemes=len(dataset.phone_map) + 1,
            num_singers=len(dataset.singer_map),
            num_languages=len(dataset.language_map)
        ).to(device)
        
        # Load checkpoint
        if args.checkpoint:
            utils.load_model_params(args.checkpoint, model, device)
        else:
            print("No checkpoint provided for inference. Using random weights.")
        
        # Run inference
        inference_svs(config, model, args.input_dir, args.output_dir)

if __name__ == '__main__':
    main()