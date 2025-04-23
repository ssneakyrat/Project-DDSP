import os
import torch
import numpy as np
import soundfile as sf  # Faster replacement for librosa.load
import torch.multiprocessing as mp  # Use torch's multiprocessing for CUDA compatibility
import torchaudio  # For GPU-accelerated audio processing
import pickle
import glob
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import defaultdict
import traceback
import logging
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
import parselmouth  # Faster F0 extraction than librosa.pyin

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dataset_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SingingVoiceDataset")

# Global constants (same as original)
DATASET_DIR = "./datasets"
CACHE_DIR = "./cache"
SAMPLE_RATE = 24000
CONTEXT_WINDOW_SEC = 2
CONTEXT_WINDOW_SAMPLES = SAMPLE_RATE * CONTEXT_WINDOW_SEC
HOP_LENGTH = 240
WIN_LENGTH = 1024
N_MELS = 80
FMIN = 40
FMAX = 12000
MIN_PHONE = 5
MIN_DURATION_MS = 10
ENABLE_ALIGNMENT_PLOTS = False

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
    audio: np.ndarray
    sr: int
    phones: List[str]
    phone_indices: List[int]
    start_times: List[int]
    end_times: List[int]
    durations: List[int]
    audio_length: int
    audio_duration_sec: float
    phone_counts: Dict[str, int]

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

# Stage 1: File gathering and initial processing
def stage1_process_file(file_metadata, phone_map, sample_rate):
    """Process a single audio/lab file pair and perform initial processing."""
    try:
        # Extract phones and timing information
        phones = []
        start_times = []
        end_times = []
        
        with open(file_metadata.lab_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    start, end, phone = parts
                    start_times.append(int(start))
                    end_times.append(int(end))
                    phones.append(phone)
        
        # Skip files with fewer phones than MIN_PHONE
        if len(phones) < MIN_PHONE:
            return None
        
        # Load audio with soundfile (much faster than librosa)
        audio, sr = sf.read(file_metadata.wav_file, dtype='float32')
        
        # Convert to mono if stereo
        if len(audio.shape) > 1 and audio.shape[1] > 1:
            audio = audio.mean(axis=1)
        
        # Resample if needed
        if sr != sample_rate:
            # Use torchaudio for faster resampling
            audio_tensor = torch.FloatTensor(audio).unsqueeze(0)
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=sample_rate
            )
            audio_tensor = resampler(audio_tensor)
            audio = audio_tensor.squeeze(0).numpy()
            sr = sample_rate
        
        audio_length = len(audio)
        audio_duration_sec = audio_length / sr
        
        # Phone statistics
        phone_counts = defaultdict(int)
        for phone in phones:
            phone_counts[phone] += 1
        
        # Calculate durations
        durations = [end - start for start, end in zip(start_times, end_times)]
        
        # Adjust durations if they are too short (same logic as original)
        min_duration_samples = int(MIN_DURATION_MS * 10000)
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
        
        if len(start_times) > 0:
            max_time = max(end_times)
            
            # Scale the timestamps to match the audio length
            start_times = [int(t * audio_length / max_time) for t in start_times]
            end_times = [int(t * audio_length / max_time) for t in end_times]
        
        # Convert phones to indices
        phone_indices = [phone_map.get(p, 0) for p in phones]
        
        # Return preprocessed data
        return AudioData(
            metadata=file_metadata,
            audio=audio,
            sr=sr,
            phones=phones,
            phone_indices=phone_indices,
            start_times=start_times,
            end_times=end_times,
            durations=durations,
            audio_length=audio_length,
            audio_duration_sec=audio_duration_sec,
            phone_counts=phone_counts
        )
            
    except Exception as e:
        logger.error(f"Error processing {file_metadata.lab_file}: {str(e)}\n{traceback.format_exc()}")
        return None

# Wrapper function for multiprocessing
def process_file_for_mp(args):
    """Wrapper function that can be pickled for multiprocessing."""
    file_metadata, phone_map, sample_rate = args
    return stage1_process_file(file_metadata, phone_map, sample_rate)

# Stage 2: GPU-based feature extraction
def stage2_extract_features_batch(batch_data, hop_length, win_length, n_mels, fmin, fmax, device, n_harmonics=10):
    """
    Process a batch of audio files on GPU for feature extraction.
    This runs in a single process to maximize GPU utilization.
    
    UPDATED: Now includes:
    - phone_seq_mel: phones aligned with mel frames
    - f0_audio: f0 aligned with audio frames
    - amplitudes: harmonic amplitudes aligned with mel frames
    """
    results = []
    
    # Initialize GPU transforms
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=win_length,
        win_length=win_length,
        hop_length=hop_length,
        f_min=fmin,
        f_max=fmax,
        n_mels=n_mels,
        power=2.0
    ).to(device)
    
    # Process each file in the batch
    for audio_data in batch_data:
        if audio_data is None:
            continue
            
        metadata = audio_data.metadata
        audio = audio_data.audio
        audio_length = audio_data.audio_length
        phone_indices = audio_data.phone_indices
        start_times = audio_data.start_times
        end_times = audio_data.end_times
        context_window_samples = CONTEXT_WINDOW_SAMPLES
        
        segments = []
        
        # Convert audio to torch tensor
        audio_tensor = torch.FloatTensor(audio).to(device)
        
        # Extract F0 using Parselmouth (much faster than librosa.pyin)
        # Run on CPU as it's not GPU accelerated
        f0 = extract_f0_parselmouth(audio, SAMPLE_RATE, hop_length)
        f0_tensor = torch.FloatTensor(f0).to(device)
        
        # NEW: Extract harmonic amplitudes based on F0
        harmonic_amplitudes = extract_harmonic_amplitudes(
            audio,
            f0,
            SAMPLE_RATE, 
            hop_length, 
            n_harmonics,
            win_length
        )
        
        # Calculate the exact expected F0 length for consistency
        target_f0_length = context_window_samples // hop_length + 1
        
        # Extract mel spectrogram
        if len(audio_tensor.shape) == 1:
            audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension
            
        # Process audio in context windows
        if audio_length < context_window_samples:
            # Pad audio if shorter than context window
            padded_audio = torch.nn.functional.pad(
                audio_tensor, (0, context_window_samples - audio_length)
            )
            
            # Extract mel spectrogram
            mel_spec = mel_transform(padded_audio)
            mel_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
            mel_norm = normalize_mel(mel_db)
            
            # Move mel to CPU and convert to numpy
            mel_np = mel_norm.squeeze(0).cpu().numpy().T
            
            # Create phone sequence aligned with audio samples
            phone_seq = np.zeros(context_window_samples)
            for p, start, end in zip(phone_indices, start_times, end_times):
                phone_seq[start:end] = p
            
            # Ensure consistent F0 length
            f0_padded = np.zeros(target_f0_length)
            f0_len = min(len(f0), target_f0_length)
            f0_padded[:f0_len] = f0[:f0_len]
            
            # NEW: Ensure consistent harmonic amplitudes length
            amplitudes_padded = np.zeros((target_f0_length, n_harmonics))
            amplitudes_len = min(len(harmonic_amplitudes), target_f0_length)
            amplitudes_padded[:amplitudes_len] = harmonic_amplitudes[:amplitudes_len]
            
            # Create phone sequence aligned with mel frames (downsampling)
            phone_seq_mel = np.zeros(target_f0_length)
            for i in range(target_f0_length):
                start_idx = i * hop_length
                end_idx = min(start_idx + hop_length, context_window_samples)
                if start_idx < context_window_samples:
                    # Use majority vote to determine phone for this frame
                    unique_phones, counts = np.unique(phone_seq[start_idx:end_idx], return_counts=True)
                    if len(unique_phones) > 0:  # Make sure there's at least one phone
                        phone_seq_mel[i] = unique_phones[np.argmax(counts)]
            
            # Create F0 aligned with audio frames (upsampling)
            t_mel = np.arange(target_f0_length) * hop_length / SAMPLE_RATE  # Time in seconds for each mel frame
            t_audio = np.arange(context_window_samples) / SAMPLE_RATE      # Time in seconds for each audio sample
            if len(f0_padded) > 0:
                # Use linear interpolation to upsample F0 to audio sample rate
                f0_audio = np.interp(t_audio, t_mel, f0_padded)
            else:
                f0_audio = np.zeros(context_window_samples)
            
            segments.append({
                'audio': audio_data.audio,
                'phone_seq': phone_seq,           # Aligned with audio frames
                'phone_seq_mel': phone_seq_mel,   # Aligned with mel frames
                'f0': f0_padded,                  # Aligned with mel frames
                'f0_audio': f0_audio,             # Aligned with audio frames
                'mel': mel_np,                    # Aligned with mel frames
                'amplitudes': amplitudes_padded,  # NEW: Harmonic amplitudes aligned with mel frames
                'singer_id': metadata.singer_idx,
                'language_id': metadata.language_idx,
                'filename': f"{metadata.singer_id}_{metadata.language_id}_{metadata.base_name}"
            })
        else:
            # Process longer audio in chunks
            for i in range(0, audio_length, context_window_samples):
                if i + context_window_samples > audio_length:
                    break
                
                # Extract audio segment
                segment_audio = audio[i:i+context_window_samples]
                segment_tensor = torch.FloatTensor(segment_audio).unsqueeze(0).to(device)
                
                # Extract mel spectrogram for this segment
                segment_mel = mel_transform(segment_tensor)
                segment_mel_db = torchaudio.transforms.AmplitudeToDB()(segment_mel)
                segment_mel_norm = normalize_mel(segment_mel_db)
                segment_mel_np = segment_mel_norm.squeeze(0).cpu().numpy().T
                
                # Ensure consistent F0 length for each segment
                f0_start_idx = i // hop_length
                f0_end_idx = f0_start_idx + target_f0_length
                
                segment_f0 = np.zeros(target_f0_length)
                if f0_start_idx < len(f0):
                    actual_f0_len = min(len(f0) - f0_start_idx, target_f0_length)
                    segment_f0[:actual_f0_len] = f0[f0_start_idx:f0_start_idx + actual_f0_len]
                
                # NEW: Extract harmonic amplitudes for this segment
                amplitudes_start_idx = i // hop_length
                amplitudes_end_idx = amplitudes_start_idx + target_f0_length
                
                segment_amplitudes = np.zeros((target_f0_length, n_harmonics))
                if amplitudes_start_idx < len(harmonic_amplitudes):
                    actual_amplitudes_len = min(len(harmonic_amplitudes) - amplitudes_start_idx, target_f0_length)
                    segment_amplitudes[:actual_amplitudes_len] = harmonic_amplitudes[amplitudes_start_idx:amplitudes_start_idx + actual_amplitudes_len]
                
                # Create phone sequence for this segment (aligned with audio frames)
                phone_seq = np.zeros(context_window_samples)
                for p, start, end in zip(phone_indices, start_times, end_times):
                    seg_start = max(0, start - i)
                    seg_end = min(context_window_samples, end - i)
                    if seg_end > seg_start and seg_start < context_window_samples:
                        phone_seq[seg_start:seg_end] = p
                
                # Create phone sequence aligned with mel frames (downsampling)
                phone_seq_mel = np.zeros(target_f0_length)
                for j in range(target_f0_length):
                    start_idx = j * hop_length
                    end_idx = min(start_idx + hop_length, context_window_samples)
                    if start_idx < context_window_samples:
                        # Use majority vote to determine phone for this frame
                        unique_phones, counts = np.unique(phone_seq[start_idx:end_idx], return_counts=True)
                        if len(unique_phones) > 0:  # Make sure there's at least one phone
                            phone_seq_mel[j] = unique_phones[np.argmax(counts)]
                
                # Create F0 aligned with audio frames (upsampling)
                t_mel = np.arange(target_f0_length) * hop_length / SAMPLE_RATE
                t_audio = np.arange(context_window_samples) / SAMPLE_RATE
                if len(segment_f0) > 0:
                    # Use linear interpolation to upsample F0 to audio sample rate
                    f0_audio = np.interp(t_audio, t_mel, segment_f0)
                else:
                    f0_audio = np.zeros(context_window_samples)
                
                segments.append({
                    'audio': segment_audio,
                    'phone_seq': phone_seq,
                    'phone_seq_mel': phone_seq_mel,
                    'f0': segment_f0,
                    'f0_audio': f0_audio,
                    'mel': segment_mel_np,
                    'amplitudes': segment_amplitudes,  # NEW: Harmonic amplitudes aligned with mel frames
                    'singer_id': metadata.singer_idx,
                    'language_id': metadata.language_idx,
                    'filename': f"{metadata.singer_id}_{metadata.language_id}_{metadata.base_name}_{i}"
                })
                
        # Return processed data
        results.append(ProcessedFeatures(
            metadata=metadata,
            segments=segments,
            phone_counts=audio_data.phone_counts,
            audio_duration_sec=audio_data.audio_duration_sec
        ))
        
    return results

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
    
    # Replace NaN with 0 for consistency with original code
    pitch_values = np.nan_to_num(pitch_values)
    
    return pitch_values

# Stage 3: Post-processing worker
def stage3_post_process(processed_features):
    """Final post-processing and statistics collection."""
    singer_language_stats = defaultdict(lambda: defaultdict(int))
    singer_duration = defaultdict(float)
    language_duration = defaultdict(float)
    phone_language_stats = defaultdict(lambda: defaultdict(int))
    
    all_segments = []
    
    for features in processed_features:
        if features is None or not features.segments:
            continue
            
        metadata = features.metadata
        
        # Add segments to final dataset
        all_segments.extend(features.segments)
        
        # Update statistics
        singer_language_stats[metadata.singer_id][metadata.language_id] += 1
        singer_duration[metadata.singer_id] += features.audio_duration_sec
        language_duration[metadata.language_id] += features.audio_duration_sec
        
        # Update phone statistics
        for phone, count in features.phone_counts.items():
            phone_language_stats[metadata.language_id][phone] += count
    
    return {
        'segments': all_segments,
        'singer_language_stats': dict(singer_language_stats),
        'singer_duration': dict(singer_duration),
        'language_duration': dict(language_duration),
        'phone_language_stats': dict(phone_language_stats)
    }

class SingingVoiceDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir=DATASET_DIR, cache_dir=CACHE_DIR, sample_rate=SAMPLE_RATE,
                 context_window_samples=CONTEXT_WINDOW_SAMPLES, rebuild_cache=False, max_files=None,
                 n_mels=N_MELS, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, fmin=FMIN, fmax=FMAX,
                 num_workers=8, batch_size=32, device='cuda' if torch.cuda.is_available() else 'cpu',
                 max_audio_length=None, max_mel_frames=None, n_harmonics=10):
        self.dataset_dir = dataset_dir
        self.cache_dir = cache_dir
        self.sample_rate = sample_rate
        self.context_window_samples = context_window_samples
        self.max_files = max_files
        self.n_harmonics = n_harmonics
        
        # Parameters for mel spectrogram extraction
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.win_length = win_length
        self.fmin = fmin
        self.fmax = fmax
        
        # Multiprocessing parameters
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.device = device
        
        # NEW: Fixed padding lengths
        self.max_audio_length = max_audio_length
        self.max_mel_frames = max_mel_frames

        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"singing_voice_multi_singer_data_{sample_rate}hz_{n_mels}mels.pkl")
        
        if os.path.exists(cache_path) and not rebuild_cache:
            self.load_cache(cache_path)
        else:
            self.build_dataset_pipeline()
            self.save_cache(cache_path)
            self.generate_distribution_log()
    
        # NEW: Calculate fixed lengths if not provided
        if self.max_audio_length is None or self.max_mel_frames is None:
            self._calculate_max_lengths()
        
    def _calculate_max_lengths(self):
        """Calculate the maximum lengths for audio and mel frames from the dataset."""
        if self.max_audio_length is None:
            # Default to context window size or calculate from data
            audio_lengths = [len(item['audio']) for item in self.data]
            self.max_audio_length = max(audio_lengths) if audio_lengths else self.context_window_samples
            
        if self.max_mel_frames is None:
            # Calculate from data or use expected size for context window
            mel_frames = [item['mel'].shape[0] for item in self.data]
            self.max_mel_frames = max(mel_frames) if mel_frames else (self.context_window_samples // self.hop_length + 1)

    def build_dataset_pipeline(self):
        """Build dataset using multi-stage pipeline approach."""
        logger.info("Building dataset using multi-stage pipeline...")
        
        # Stage 1: Scan directory and collect metadata
        self.scan_dataset_structure()
        
        # Create file processing tasks
        processing_tasks = self.create_processing_tasks()
        
        if self.max_files and self.max_files < len(processing_tasks):
            logger.info(f"Limiting dataset to {self.max_files} files out of {len(processing_tasks)}")
            processing_tasks = processing_tasks[:self.max_files]
        
        # Initialize results
        self.data = []
        self.singer_language_stats = defaultdict(lambda: defaultdict(int))
        self.singer_duration = defaultdict(float)
        self.language_duration = defaultdict(float)
        self.phone_language_stats = defaultdict(lambda: defaultdict(int))
        
        # Stage 1: Multi-process file loading and initial processing
        logger.info("Stage 1: Loading and preprocessing files...")
        preprocessed_data = self.run_stage1_preprocessing(processing_tasks)
        
        # Group data into batches for GPU processing
        batches = [preprocessed_data[i:i+self.batch_size] 
                  for i in range(0, len(preprocessed_data), self.batch_size)]
        
        # Stage 2: GPU feature extraction (single process)
        logger.info(f"Stage 2: Extracting features on {self.device}...")
        processed_features = []
        
        for batch_idx, batch in enumerate(tqdm(batches, desc="Processing batches on GPU")):
            batch_results = stage2_extract_features_batch(
                batch, self.hop_length, self.win_length, 
                self.n_mels, self.fmin, self.fmax, self.device,
                n_harmonics=self.n_harmonics
            )
            processed_features.extend(batch_results)
        
        # Stage 3: Final post-processing and collection
        logger.info("Stage 3: Post-processing and collecting results...")
        results = stage3_post_process(processed_features)
        
        # Update dataset with results
        self.data = results['segments']
        
        # Update statistics (convert defaultdicts to regular dicts for pickling)
        self.singer_language_stats = results['singer_language_stats']
        self.singer_duration = results['singer_duration']
        self.language_duration = results['language_duration']
        self.phone_language_stats = results['phone_language_stats']
        
        logger.info(f"Dataset built with {len(self.data)} segments")
    
    def scan_dataset_structure(self):
        """Scan directory structure and create mappings."""
        # Verify dataset directory exists
        if not os.path.exists(self.dataset_dir):
            raise ValueError(f"Dataset directory does not exist: {self.dataset_dir}")
            
        # Find all singer directories
        singer_dirs = [d for d in glob.glob(os.path.join(self.dataset_dir, "*")) if os.path.isdir(d)]
        logger.info(f"Found {len(singer_dirs)} singers: {[os.path.basename(d) for d in singer_dirs]}")
        
        if not singer_dirs:
            raise ValueError(f"No singer directories found in {self.dataset_dir}")
            
        # Create singer ID mapping
        self.singer_map = {os.path.basename(s): i for i, s in enumerate(sorted(singer_dirs))}
        self.inv_singer_map = {i: s for s, i in self.singer_map.items()}
        
        # Find all language directories and create language mapping
        language_dirs = []
        for singer_dir in singer_dirs:
            lang_dirs = [d for d in glob.glob(os.path.join(singer_dir, "*")) if os.path.isdir(d)]
            language_dirs.extend(lang_dirs)
        
        unique_languages = set(os.path.basename(l) for l in language_dirs)
        self.language_map = {lang: i for i, lang in enumerate(sorted(unique_languages))}
        self.inv_language_map = {i: lang for lang, i in self.language_map.items()}
        
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
        self.phone_map = {phone: i+1 for i, phone in enumerate(sorted(all_phones))}
        self.inv_phone_map = {i: phone for phone, i in self.phone_map.items()}
        logger.info(f"Found {len(self.phone_map)} unique phones")
    
    def create_processing_tasks(self):
        """Create list of file processing tasks."""
        tasks = []
        
        for singer_dir in glob.glob(os.path.join(self.dataset_dir, "*")):
            if not os.path.isdir(singer_dir):
                continue
                
            singer_id = os.path.basename(singer_dir)
            singer_idx = self.singer_map[singer_id]
            
            for lang_dir in glob.glob(os.path.join(singer_dir, "*")):
                if not os.path.isdir(lang_dir):
                    continue
                    
                language_id = os.path.basename(lang_dir)
                language_idx = self.language_map[language_id]
                
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
    
    def run_stage1_preprocessing(self, tasks):
        """Run Stage 1: Multiprocessing for file loading and initial processing."""
        # Process files in parallel
        print( 'num worker', self.num_workers)
        max_workers = self.num_workers if self.num_workers > 0 else min(32, os.cpu_count() + 4)
        
        logger.info(f"Stage 1: Processing files with {max_workers} workers")
        
        # Prepare args for multiprocessing - must be picklable
        mp_args = [(task, self.phone_map, self.sample_rate) for task in tasks]
        
        # Use multiprocessing to process files
        with mp.Pool(max_workers) as pool:
            results = list(tqdm(
                pool.imap(process_file_for_mp, mp_args),
                total=len(tasks),
                desc="Preprocessing files"
            ))
        
        # Filter out None results
        valid_results = [r for r in results if r is not None]
        logger.info(f"Successfully preprocessed {len(valid_results)} files out of {len(tasks)}")
        
        return valid_results
    
    def save_cache(self, cache_path):
        cache_data = {
            'data': self.data,
            'phone_map': self.phone_map,
            'inv_phone_map': self.inv_phone_map,
            'singer_map': self.singer_map,
            'inv_singer_map': self.inv_singer_map,
            'language_map': self.language_map,
            'inv_language_map': self.inv_language_map,
            'singer_language_stats': self.singer_language_stats,
            'singer_duration': self.singer_duration,
            'language_duration': self.language_duration,
            'phone_language_stats': self.phone_language_stats
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        logger.info(f"Dataset cached to {cache_path}")
    
    def load_cache(self, cache_path):
        logger.info(f"Loading dataset from cache: {cache_path}")
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        self.data = cache_data['data']
        self.phone_map = cache_data['phone_map']
        self.inv_phone_map = cache_data['inv_phone_map']
        self.singer_map = cache_data['singer_map']
        self.inv_singer_map = cache_data['inv_singer_map']
        self.language_map = cache_data['language_map']
        self.inv_language_map = cache_data['inv_language_map']
        
        # Load statistics if available
        self.singer_language_stats = cache_data.get('singer_language_stats', {})
        self.singer_duration = cache_data.get('singer_duration', {})
        self.language_duration = cache_data.get('language_duration', {})
        self.phone_language_stats = cache_data.get('phone_language_stats', {})
        
        logger.info(f"Loaded {len(self.data)} segments with {len(self.singer_map)} singers, "
                  f"{len(self.language_map)} languages, and {len(self.phone_map)} unique phones")
    
    def generate_distribution_log(self):
        """Generate a log file with dataset distribution statistics."""
        log_path = os.path.join(self.cache_dir, "dataset_distribution.txt")
        
        with open(log_path, 'w') as f:
            f.write("=== SINGING VOICE DATASET DISTRIBUTION ===\n\n")
            
            # Overall statistics
            f.write(f"Total segments: {len(self.data)}\n")
            f.write(f"Total singers: {len(self.singer_map)}\n")
            f.write(f"Total languages: {len(self.language_map)}\n")
            f.write(f"Total unique phones: {len(self.phone_map)}\n\n")
            
            # Singer statistics
            f.write("=== SINGER STATISTICS ===\n")
            for singer_id in self.singer_map:
                count = sum(self.singer_language_stats.get(singer_id, {}).values())
                duration = self.singer_duration.get(singer_id, 0)
                f.write(f"Singer {singer_id} (ID: {self.singer_map[singer_id]}):\n")
                f.write(f"  - Files: {count}\n")
                f.write(f"  - Total duration: {duration:.2f} seconds\n")
                langs = self.singer_language_stats.get(singer_id, {}).keys()
                f.write(f"  - Languages: {', '.join(langs)}\n\n")
            
            # Language statistics
            f.write("=== LANGUAGE STATISTICS ===\n")
            for language_id in self.language_map:
                singer_count = sum(1 for s in self.singer_language_stats 
                                 if language_id in self.singer_language_stats.get(s, {}))
                
                file_count = sum(self.singer_language_stats.get(s, {}).get(language_id, 0) 
                                for s in self.singer_language_stats)
                
                duration = self.language_duration.get(language_id, 0)
                f.write(f"Language {language_id} (ID: {self.language_map[language_id]}):\n")
                f.write(f"  - Files: {file_count}\n")
                f.write(f"  - Singers: {singer_count}\n")
                f.write(f"  - Total duration: {duration:.2f} seconds\n")
                
                # Phone distribution for this language
                if language_id in self.phone_language_stats:
                    f.write(f"  - Phone distribution:\n")
                    phone_counts = self.phone_language_stats[language_id]
                    sorted_phones = sorted(phone_counts.items(), key=lambda x: x[1], reverse=True)
                    for phone, count in sorted_phones:
                        f.write(f"    {phone}: {count}\n")
                f.write("\n")
        
        logger.info(f"Distribution log written to {log_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Extract original tensors
        audio = torch.FloatTensor(item['audio'])
        phone_seq = torch.LongTensor(item['phone_seq'])                  # Aligned with audio
        phone_seq_mel = torch.LongTensor(item['phone_seq_mel'])          # Aligned with mel frames
        f0 = torch.FloatTensor(item['f0'])                               # Aligned with mel frames
        f0_audio = torch.FloatTensor(item['f0_audio'])                   # Aligned with audio frames  
        mel = torch.FloatTensor(item['mel'])
        singer_id = torch.LongTensor([item['singer_id']])
        language_id = torch.LongTensor([item['language_id']])
        
        # Load harmonic amplitudes
        amplitudes = torch.FloatTensor(item['amplitudes'])
        
        # NEW: Pad tensors to fixed lengths
        # 1. Pad audio-aligned tensors
        audio_length = audio.size(0)
        if audio_length < self.max_audio_length:
            pad_length = self.max_audio_length - audio_length
            audio = F.pad(audio, (0, pad_length))
            phone_seq = F.pad(phone_seq, (0, pad_length))
            f0_audio = F.pad(f0_audio, (0, pad_length))
        
        # 2. Pad mel-aligned tensors
        mel_frames = mel.size(0)
        if mel_frames < self.max_mel_frames:
            pad_frames = self.max_mel_frames - mel_frames
            # For 2D tensors (mel)
            mel = F.pad(mel, (0, 0, 0, pad_frames))
            # For 1D tensors
            phone_seq_mel = F.pad(phone_seq_mel, (0, pad_frames))
            f0 = F.pad(f0, (0, pad_frames))
            # For 2D amplitudes
            amplitudes = F.pad(amplitudes, (0, 0, 0, pad_frames))
        
        # Create one-hot encodings
        phone_one_hot = F.one_hot(phone_seq.long(), num_classes=len(self.phone_map)+1).float()
        phone_mel_one_hot = F.one_hot(phone_seq_mel.long(), num_classes=len(self.phone_map)+1).float()
        singer_one_hot = F.one_hot(singer_id.long(), num_classes=len(self.singer_map)).float().squeeze(0)
        language_one_hot = F.one_hot(language_id.long(), num_classes=len(self.language_map)).float().squeeze(0)
        
        return {
            'audio': audio,
            'phone_seq': phone_seq,
            'phone_seq_mel': phone_seq_mel,
            'phone_one_hot': phone_one_hot,
            'phone_mel_one_hot': phone_mel_one_hot,
            'f0': f0,
            'f0_audio': f0_audio,
            'mel': mel,
            'singer_id': singer_id,
            'language_id': language_id,
            'singer_one_hot': singer_one_hot,
            'language_one_hot': language_one_hot,
            'amplitudes': amplitudes,
            'filename': item['filename']
        }

def get_dataloader(batch_size=16, num_workers=4, pin_memory=True, persistent_workers=True, 
                 max_files=None, device='cuda', collate_fn=None, shuffle=True, n_harmonics=10):
    """
    Get a dataloader for the singing voice dataset.
    
    Args:
        batch_size: Number of samples per batch
        num_workers: Number of workers for the DataLoader
        pin_memory: Whether to pin memory in DataLoader
        persistent_workers: Whether to keep workers alive between epochs
        max_files: Maximum number of files to process
        device: Device to use for GPU acceleration ('cuda' or 'cpu')
        n_harmonics: Number of harmonics to extract (NEW)
    """
    dataset = SingingVoiceDataset(
        rebuild_cache=False, 
        max_files=max_files,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        fmin=FMIN,
        fmax=FMAX,
        num_workers=num_workers,
        device=device,
        n_harmonics=n_harmonics,  # NEW: Pass n_harmonics parameter
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        collate_fn=collate_fn
    )
    return dataloader, dataset

if __name__ == "__main__":
    # Example usage
    loader, dataset = get_dataloader(
        batch_size=16, 
        num_workers=4,
        max_files=100  # Limit for testing
    )
    
    # Print dataset stats
    print(f"Dataset size: {len(dataset)}")
    
    # Benchmark loading time
    start_time = time.time()
    for i, batch in enumerate(tqdm(loader, desc="Loading batches")):
        if i > 10:  # Load a few batches for benchmark
            break
    print(f"Time to load 10 batches: {time.time() - start_time:.2f} seconds")