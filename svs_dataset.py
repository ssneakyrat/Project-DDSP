import torch
import torch.nn.functional as F
import numpy as np

def svs_collate_fn(batch):
    """
    Custom collate function for SVS dataset that handles variable-length tensors.
    """
    # Get batch size
    batch_size = len(batch)
    
    # Initialize output dictionary with lists
    collated = {
        'name': [],
    }
    
    # Find max lengths for various sequence fields
    max_audio_len = max(item['audio'].size(0) for item in batch)
    max_phoneme_len = max(item['phonemes'].size(0) for item in batch)
    max_f0_len = max(item['f0'].size(0) for item in batch)
    max_duration_len = max(item['durations'].size(0) for item in batch)
    
    # Initialize tensors with proper shapes
    collated['audio'] = torch.zeros(batch_size, max_audio_len)
    collated['phonemes'] = torch.zeros(batch_size, max_phoneme_len, dtype=torch.long)
    collated['f0'] = torch.zeros(batch_size, max_f0_len)
    collated['durations'] = torch.zeros(batch_size, max_duration_len)
    
    # Process singer_id and language_id (these should be single values per sample)
    collated['singer_id'] = torch.zeros(batch_size, dtype=torch.long)
    collated['language_id'] = torch.zeros(batch_size, dtype=torch.long)
    
    # If mel spectrograms are included
    if 'mel' in batch[0]:
        mel_dim = batch[0]['mel'].size(-1)
        max_mel_len = max(item['mel'].size(0) for item in batch)
        collated['mel'] = torch.zeros(batch_size, max_mel_len, mel_dim)
    
    # Fill the tensors with actual data
    for i, item in enumerate(batch):
        collated['name'].append(item['name'])
        
        # Handle variable-length sequences with proper padding
        audio_len = item['audio'].size(0)
        phoneme_len = item['phonemes'].size(0)
        f0_len = item['f0'].size(0)
        duration_len = item['durations'].size(0)
        
        collated['audio'][i, :audio_len] = item['audio']
        collated['phonemes'][i, :phoneme_len] = item['phonemes']
        collated['f0'][i, :f0_len] = item['f0']
        collated['durations'][i, :duration_len] = item['durations']
        
        # Handle scalar values
        collated['singer_id'][i] = item['singer_id']
        collated['language_id'][i] = item['language_id']
        
        # Handle mel spectrograms if present
        if 'mel' in item:
            mel_len = item['mel'].size(0)
            collated['mel'][i, :mel_len] = item['mel']
    
    return collated

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
        item = self.dataset[idx]
        
        # Ensure consistent types and shapes
        if isinstance(item['singer_id'], list) or isinstance(item['singer_id'], np.ndarray):
            singer_id = torch.tensor(item['singer_id'][0], dtype=torch.long)
        else:
            singer_id = torch.tensor(item['singer_id'], dtype=torch.long)
        
        if isinstance(item['language_id'], list) or isinstance(item['language_id'], np.ndarray):
            language_id = torch.tensor(item['language_id'][0], dtype=torch.long)
        else:
            language_id = torch.tensor(item['language_id'], dtype=torch.long)
        
        # Construct the required output format
        return {
            'name': item['filename'],
            'audio': item['audio'],
            'phonemes': torch.tensor(item['phone_seq'], dtype=torch.long),
            'durations': self.calculate_durations(torch.tensor(item['phone_seq'])),
            'f0': torch.tensor(item['f0'], dtype=torch.float),
            'singer_id': singer_id,
            'language_id': language_id,
            'mel': torch.tensor(item['mel'], dtype=torch.float)
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
        rebuild_cache=False,
        max_files=100
    )
    
    # Create validation dataset
    valid_dataset = dataset_cls(
        dataset_dir=args.data.valid_path,
        cache_dir="./cache/valid",
        sample_rate=args.data.sampling_rate,
        rebuild_cache=False,
        max_files=10
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
        drop_last=True,
        persistent_workers=True,
        collate_fn=svs_collate_fn  # Use the custom collate function
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset_wrapped,
        batch_size=args.inference.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=svs_collate_fn  # Use the custom collate function
    )
    
    return train_loader, valid_loader