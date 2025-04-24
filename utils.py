import os
import torch
import numpy as np
import soundfile as sf
import torchaudio
import parselmouth
import logging
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dataset_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SingingVoiceUtils")

# Define dataclass for holding file metadata
@dataclass
class FileMetadata:
    lab_file: str
    wav_file: str
    singer_id: str
    language_id: str
    singer_idx: int
    language_idx: int
    base_name: str

# Define dataclass for preprocessed audio data
@dataclass
class AudioData:
    metadata: FileMetadata
    audio: Optional[np.ndarray]
    sr: int
    phones: List[str]
    phone_indices: List[int]
    start_times: List[int]
    end_times: List[int]
    durations: List[int]
    audio_length: int
    audio_duration_sec: float
    phone_counts: Dict[str, int]
    chunks: List[Dict[str, Any]]

# Define dataclass for processed features
@dataclass
class ProcessedFeatures:
    metadata: FileMetadata
    segments: List[Dict[str, Any]]
    phone_counts: Dict[str, int]
    audio_duration_sec: float

def extract_harmonic_amplitudes(audio, f0, sample_rate, hop_length, n_harmonics, window_length=1024):
    """
    Extract harmonic amplitudes from audio based on F0 values.
    
    Args:
        audio (np.ndarray): Audio signal
        f0 (np.ndarray): F0 values aligned with frames (hop_length spacing)
        sample_rate (int): Sample rate of audio
        hop_length (int): Hop length for frame alignment
        n_harmonics (int): Number of harmonics to extract
        window_length (int): Window length for STFT
        
    Returns:
        np.ndarray: Harmonic amplitudes with shape [time_frames, n_harmonics]
    """
    # Convert audio to tensor for STFT calculation
    audio_tensor = torch.FloatTensor(audio).unsqueeze(0)
    
    # Calculate STFT for spectral analysis
    stft = torch.stft(
        audio_tensor, 
        n_fft=window_length, 
        hop_length=hop_length, 
        win_length=window_length, 
        window=torch.hann_window(window_length), 
        return_complex=True
    )
    
    # Convert to magnitude spectrum
    mag_spec = torch.abs(stft)[0].cpu().numpy()  # [freq_bins, time_frames]
    
    # Transpose to [time_frames, freq_bins] for easier processing
    mag_spec = mag_spec.T
    
    # Create frequency axis
    freq_bins = mag_spec.shape[1]
    freqs = np.linspace(0, sample_rate/2, freq_bins)
    
    # Initialize array for harmonic amplitudes
    time_frames = len(f0)
    amplitudes = np.zeros((time_frames, n_harmonics))
    
    # For each time frame
    for t in range(min(time_frames, mag_spec.shape[0])):
        # Skip unvoiced or invalid frames
        if f0[t] <= 0:
            continue
            
        # For each harmonic
        for h in range(n_harmonics):
            # Calculate frequency of this harmonic
            harmonic_freq = f0[t] * (h + 1)  # h+1 because first harmonic is 1*f0
            
            # Skip if harmonic is above Nyquist frequency
            if harmonic_freq >= sample_rate/2:
                continue
                
            # Find closest frequency bin
            bin_idx = np.argmin(np.abs(freqs - harmonic_freq))
            
            # Extract amplitude at this bin
            amplitudes[t, h] = mag_spec[t, bin_idx]
    
    # Normalize amplitudes per frame (make sum=1 for each frame)
    for t in range(time_frames):
        if np.sum(amplitudes[t]) > 0:
            amplitudes[t] = amplitudes[t] / np.sum(amplitudes[t])
    
    return amplitudes

def normalize_mel(mel_spec):
    """Normalize mel spectrogram."""
    mel_spec = mel_spec - mel_spec.min()
    mel_spec = mel_spec / (mel_spec.max() + 1e-8)
    return mel_spec

def extract_f0_parselmouth(audio, sample_rate, hop_length):
    """Extract F0 using Parselmouth (Praat)."""
    # Create a Praat Sound object
    sound = parselmouth.Sound(values=audio, sampling_frequency=sample_rate)
    
    # Define min/max pitch frequencies (in Hz) 
    pitch_floor = 65.0  # ~C2 in Hz
    pitch_ceiling = 2093.0  # ~C7 in Hz
    
    # Extract pitch
    pitch = sound.to_pitch(
        time_step=hop_length/sample_rate,
        pitch_floor=pitch_floor,
        pitch_ceiling=pitch_ceiling
    )
    
    # Extract pitch values
    pitch_values = pitch.selected_array['frequency']
    
    # Replace unvoiced regions (0) with NaN
    pitch_values[pitch_values==0] = np.nan
    
    # Replace NaN with 0 for consistency
    pitch_values = np.nan_to_num(pitch_values)
    
    return pitch_values

def scan_directory_structure(dataset_dir):
    """
    Scan directory structure and create mappings for singers, languages, and phones.
    
    Args:
        dataset_dir (str): Path to the dataset directory
        
    Returns:
        dict: Containing singer_map, language_map, phone_map and their inverses
    """
    # Verify dataset directory exists
    if not os.path.exists(dataset_dir):
        raise ValueError(f"Dataset directory does not exist: {dataset_dir}")
    
    # Find all singer directories
    singer_dirs = [d for d in glob.glob(os.path.join(dataset_dir, "*")) if os.path.isdir(d)]
    logger.info(f"Found {len(singer_dirs)} singers: {[os.path.basename(d) for d in singer_dirs]}")
    
    if not singer_dirs:
        raise ValueError(f"No singer directories found in {dataset_dir}")
            
    # Create singer ID mapping
    singer_map = {os.path.basename(s): i for i, s in enumerate(sorted(singer_dirs))}
    inv_singer_map = {i: s for s, i in singer_map.items()}
    
    # Find all language directories and create language mapping
    language_dirs = []
    for singer_dir in singer_dirs:
        lang_dirs = [d for d in glob.glob(os.path.join(singer_dir, "*")) if os.path.isdir(d)]
        language_dirs.extend(lang_dirs)
    
    unique_languages = set(os.path.basename(l) for l in language_dirs)
    language_map = {lang: i for i, lang in enumerate(sorted(unique_languages))}
    inv_language_map = {i: lang for lang, i in language_map.items()}
    
    # Scan for all phonemes
    all_phones = set()
    
    for singer_dir in singer_dirs:
        singer_id = os.path.basename(singer_dir)
        
        for lang_dir in glob.glob(os.path.join(singer_dir, "*")):
            if not os.path.isdir(lang_dir):
                continue
                
            language_id = os.path.basename(lang_dir)
            
            # Check for lab directory
            lab_dir = os.path.join(lang_dir, "lab")
            if not os.path.exists(lab_dir):
                continue
            
            # List lab files
            lab_files = glob.glob(os.path.join(lab_dir, "*.lab"))
            
            # Read phones from files
            for lab_file in lab_files:
                try:
                    with open(lab_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) == 3:
                                _, _, phone = parts
                                all_phones.add(phone)
                except Exception as e:
                    logger.error(f"Error reading lab file {lab_file}: {str(e)}")
    
    # Create phone mapping
    phone_map = {phone: i+1 for i, phone in enumerate(sorted(all_phones))}
    inv_phone_map = {i: phone for phone, i in phone_map.items()}
    logger.info(f"Found {len(phone_map)} unique phones")
    
    return {
        'singer_map': singer_map,
        'inv_singer_map': inv_singer_map,
        'language_map': language_map,
        'inv_language_map': inv_language_map,
        'phone_map': phone_map,
        'inv_phone_map': inv_phone_map
    }

def create_file_tasks(dataset_dir, singer_map, language_map):
    """
    Create list of file processing tasks.
    
    Args:
        dataset_dir (str): Path to the dataset directory
        singer_map (dict): Mapping of singer IDs to indices
        language_map (dict): Mapping of language IDs to indices
        
    Returns:
        list: List of FileMetadata objects
    """
    tasks = []
    
    for singer_dir in glob.glob(os.path.join(dataset_dir, "*")):
        if not os.path.isdir(singer_dir):
            continue
            
        singer_id = os.path.basename(singer_dir)
        singer_idx = singer_map[singer_id]
        
        for lang_dir in glob.glob(os.path.join(singer_dir, "*")):
            if not os.path.isdir(lang_dir):
                continue
                
            language_id = os.path.basename(lang_dir)
            language_idx = language_map[language_id]
            
            # Check for lab and wav directories
            lab_dir = os.path.join(lang_dir, "lab")
            wav_dir = os.path.join(lang_dir, "wav")
            
            if not os.path.exists(lab_dir) or not os.path.exists(wav_dir):
                continue
            
            # List lab files
            lab_files = glob.glob(os.path.join(lab_dir, "*.lab"))
            
            for lab_file in lab_files:
                base_name = os.path.basename(lab_file).replace('.lab', '')
                wav_file = os.path.join(wav_dir, f"{base_name}.wav")
                
                if not os.path.exists(wav_file):
                    continue
                
                tasks.append(FileMetadata(
                    lab_file=lab_file,
                    wav_file=wav_file,
                    singer_id=singer_id,
                    language_id=language_id,
                    singer_idx=singer_idx,
                    language_idx=language_idx,
                    base_name=base_name
                ))
    
    logger.info(f"Created {len(tasks)} processing tasks")
    return tasks

def estimate_max_lengths(file_tasks, sample_rate, hop_length, max_files=100):
    """
    Estimate the maximum audio length and mel frames by scanning a subset of files.
    
    Args:
        file_tasks (list): List of FileMetadata objects
        sample_rate (int): Sample rate of audio
        hop_length (int): Hop length for mel spectrogram
        max_files (int): Maximum number of files to scan
        
    Returns:
        tuple: (max_audio_length, max_mel_frames)
    """
    logger.info(f"Estimating maximum audio length from {min(len(file_tasks), max_files)} files")
    
    # Select a subset of files to scan
    if len(file_tasks) > max_files:
        subset_tasks = random.sample(file_tasks, max_files)
    else:
        subset_tasks = file_tasks
    
    max_audio_length = 0
    
    for task in subset_tasks:
        try:
            # Get audio file info without loading all data
            info = sf.info(task.wav_file)
            file_length = int(info.frames)
            
            # Update max length
            max_audio_length = max(max_audio_length, file_length)
        except Exception as e:
            logger.warning(f"Couldn't get info from {task.wav_file}: {str(e)}")
    
    # Calculate max mel frames from max audio length
    max_mel_frames = max_audio_length // hop_length + 1
    
    # Define a standard chunk size (e.g., 5 seconds)
    import math
    chunk_size_sec = 5  # Use 5-second chunks
    chunk_size_samples = sample_rate * chunk_size_sec
    
    # Use chunk size if it's smaller than max detected, otherwise use the max
    if chunk_size_samples < max_audio_length:
        max_audio_length = chunk_size_samples
        max_mel_frames = max_audio_length // hop_length + 1
        logger.info(f"Using standard chunk size: {max_audio_length} samples ({chunk_size_sec} seconds)")
    else:
        logger.info(f"Using detected max length: {max_audio_length} samples ({max_audio_length/sample_rate:.2f} seconds)")
    
    logger.info(f"Max mel frames: {max_mel_frames}")
    
    return max_audio_length, max_mel_frames

def process_file_metadata(lab_file, phone_map, min_phone=5, min_duration_ms=10):
    """
    Process a lab file to extract phone and timing information.
    
    Args:
        lab_file (str): Path to lab file
        phone_map (dict): Mapping of phones to indices
        min_phone (int): Minimum number of phones required
        min_duration_ms (int): Minimum duration of a phone in milliseconds
        
    Returns:
        tuple: (phones, phone_indices, start_times, end_times, durations)
    """
    phones = []
    start_times = []
    end_times = []
    
    with open(lab_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                start, end, phone = parts
                start_times.append(int(start))
                end_times.append(int(end))
                phones.append(phone)
    
    # Skip files with fewer phones than MIN_PHONE
    if len(phones) < min_phone:
        return None, None, None, None, None
    
    # Calculate durations
    durations = [end - start for start, end in zip(start_times, end_times)]
    
    # Adjust durations if they are too short
    min_duration_samples = int(min_duration_ms * 10000)
    for i in range(len(durations)):
        if durations[i] < min_duration_samples:
            # Try to borrow from left neighbor
            if i > 0 and durations[i-1] > min_duration_samples * 2:
                borrow_amount = min(min_duration_samples - durations[i], durations[i-1] - min_duration_samples)
                start_times[i] -= borrow_amount
                end_times[i-1] -= borrow_amount
                durations[i] += borrow_amount
                durations[i-1] -= borrow_amount
            
            # If still too short, try to borrow from right neighbor
            if durations[i] < min_duration_samples and i < len(durations) - 1 and durations[i+1] > min_duration_samples * 2:
                borrow_amount = min(min_duration_samples - durations[i], durations[i+1] - min_duration_samples)
                end_times[i] += borrow_amount
                start_times[i+1] += borrow_amount
                durations[i] += borrow_amount
                durations[i+1] -= borrow_amount
            
            # Update the duration
            durations[i] = end_times[i] - start_times[i]
    
    # Convert phones to indices
    phone_indices = [phone_map.get(p, 0) for p in phones]
    
    return phones, phone_indices, start_times, end_times, durations

def standardized_collate_fn(batch):
    """
    Very simple collate function that just stacks tensors since they're all padded to the same size.
    
    Args:
        batch (list): List of dictionaries with tensor data
        
    Returns:
        dict: Dictionary with batched tensors
    """
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
            # All tensors should have the same shape, so just stack them
            try:
                batch_dict[key] = torch.stack([sample[key] for sample in batch])
            except RuntimeError as e:
                shapes = [sample[key].shape for sample in batch]
                logger.error(f"ERROR stacking {key} tensors. Shapes: {shapes}")
                raise e
    
    return batch_dict