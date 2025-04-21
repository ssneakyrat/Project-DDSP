import torch
import torch.nn.functional as F
import numpy as np

class SVSDatasetWrapper(torch.utils.data.Dataset):
    """
    Dataset wrapper for Singing Voice Synthesis
    
    Takes the existing dataset and transforms it to the format required by the SVS vocoder
    """
    def __init__(self, dataset):
        """
        Initialize the SVS dataset wrapper
        
        Args:
            dataset: The base dataset containing audio, phonemes, F0, singer IDs, etc.
        """
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """
        Get an item from the dataset and format it for the SVS vocoder
        
        Returns:
            Dictionary containing:
                - name: File identifier
                - audio: Audio waveform
                - phonemes: Phoneme sequence
                - durations: Phoneme durations
                - f0: Fundamental frequency
                - singer_id: Singer identifier
                - language_id: Language identifier
                - mel: Mel spectrogram (for compatibility)
        """
        item = self.dataset[idx]
        
        # Construct the required output format
        return {
            'name': item['filename'],
            'audio': item['audio'],
            'phonemes': item['phone_seq'],  # Phoneme indices sequence
            'durations': self.calculate_durations(item['phone_seq']),  # Frame durations for each phoneme
            'f0': item['f0'],  # Fundamental frequency sequence
            'singer_id': item['singer_id'],
            'language_id': item['language_id'],
            'mel': item['mel']  # Keep mel for training validation
        }
    
    def calculate_durations(self, phone_seq):
        """
        Calculate phoneme durations from phone sequence
        
        Args:
            phone_seq: Sequence of phone IDs
            
        Returns:
            durations: Tensor of frame durations for each phoneme
        """
        # This implementation depends on how phonemes are represented
        # Here's a simple approach that counts consecutive identical phonemes
        if len(phone_seq.shape) > 1:
            # If one-hot encoded, convert to indices
            if phone_seq.shape[-1] > 1:
                phone_seq = torch.argmax(phone_seq, dim=-1)
        
        # Find boundaries where phoneme changes
        boundaries = torch.where(phone_seq[:-1] != phone_seq[1:])[0] + 1
        boundaries = torch.cat([torch.tensor([0], device=phone_seq.device), 
                               boundaries, 
                               torch.tensor([len(phone_seq)], device=phone_seq.device)])
        
        # Calculate durations
        durations = boundaries[1:] - boundaries[:-1]
        
        return durations.float()

def get_svs_data_loaders(args, dataset_cls):
    """
    Create data loaders for SVS training
    
    Args:
        args: Configuration arguments
        dataset_cls: Dataset class to use
        
    Returns:
        train_loader: DataLoader for training
        valid_loader: DataLoader for validation
    """
    # Create training dataset
    train_dataset = dataset_cls(
        dataset_dir=args.data.train_path,
        cache_dir="./cache/train",
        sample_rate=args.data.sampling_rate,
        rebuild_cache=False
    )
    
    # Create validation dataset
    valid_dataset = dataset_cls(
        dataset_dir=args.data.valid_path,
        cache_dir="./cache/valid",
        sample_rate=args.data.sampling_rate,
        rebuild_cache=False
    )
    
    # Wrap datasets
    train_dataset_wrapped = SVSDatasetWrapper(train_dataset)
    valid_dataset_wrapped = SVSDatasetWrapper(valid_dataset)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset_wrapped,
        batch_size=args.train.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset_wrapped,
        batch_size=args.inference.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, valid_loader