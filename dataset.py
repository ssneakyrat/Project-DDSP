import os
import torch
import numpy as np
import soundfile as sf
import torch.multiprocessing as mp
import torchaudio
import pickle
import glob
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import defaultdict
import traceback
import logging
import time
import random
import math

# Import utilities from utils.py
from utils import (
    FileMetadata, AudioData, ProcessedFeatures,
    extract_harmonic_amplitudes, normalize_mel, extract_f0_parselmouth,
    scan_directory_structure, create_file_tasks, estimate_max_lengths,
    process_file_metadata, standardized_collate_fn
)

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

# Global constants
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

# Stage 1: File gathering and initial processing
def stage1_process_file(file_metadata, phone_map, sample_rate, max_audio_length, max_mel_frames, hop_length):
    """
    Process a single audio/lab file pair and perform initial processing.
    Now includes chunking to max_audio_length and padding of final chunk.
    """
    try:
        # Extract phones and timing information using utility function
        phones, phone_indices, start_times, end_times, durations = process_file_metadata(
            file_metadata.lab_file, phone_map, MIN_PHONE, MIN_DURATION_MS
        )
        
        if phones is None:
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
        
        if len(start_times) > 0:
            max_time = max(end_times)
            
            # Scale the timestamps to match the audio length
            start_times = [int(t * audio_length / max_time) for t in start_times]
            end_times = [int(t * audio_length / max_time) for t in end_times]
        
        # Create chunks based on max_audio_length
        # Each chunk except possibly the last will be exactly max_audio_length
        chunks = []
        for i in range(0, audio_length, max_audio_length):
            end_idx = min(i + max_audio_length, audio_length)
            chunk_audio = audio[i:end_idx]
            
            # Create phone sequence for this chunk
            chunk_phones = []
            chunk_phone_indices = []
            chunk_start_times = []
            chunk_end_times = []
            chunk_durations = []
            
            for p, p_idx, start, end in zip(phones, phone_indices, start_times, end_times):
                # Check if this phone overlaps with the current chunk
                if end > i and start < end_idx:
                    # Adjust timing to be relative to chunk start
                    chunk_start = max(0, start - i)
                    chunk_end = min(end_idx - i, end - i)
                    
                    chunk_phones.append(p)
                    chunk_phone_indices.append(p_idx)
                    chunk_start_times.append(chunk_start)
                    chunk_end_times.append(chunk_end)
                    chunk_durations.append(chunk_end - chunk_start)
            
            # Skip chunks with no phones
            if not chunk_phones:
                continue
            
            # Calculate expected mel frames for this chunk
            chunk_mel_frames = (end_idx - i) // hop_length + 1
            
            # Pad last chunk to max_audio_length if needed
            if end_idx - i < max_audio_length:
                padding_size = max_audio_length - (end_idx - i)
                padded_chunk = np.pad(chunk_audio, (0, padding_size), 'constant')
                
                # No need to adjust phone timing as they're already relative to chunk start
                
                chunks.append({
                    'audio': padded_chunk,
                    'phones': chunk_phones,
                    'phone_indices': chunk_phone_indices,
                    'start_times': chunk_start_times,
                    'end_times': chunk_end_times,
                    'durations': chunk_durations,
                    'is_padded': True,
                    'original_length': end_idx - i,
                    'mel_frames': max_mel_frames,  # Use max_mel_frames for consistency
                    'chunk_idx': len(chunks)
                })
            else:
                # Full-sized chunk, no padding needed
                chunks.append({
                    'audio': chunk_audio,
                    'phones': chunk_phones,
                    'phone_indices': chunk_phone_indices,
                    'start_times': chunk_start_times,
                    'end_times': chunk_end_times,
                    'durations': chunk_durations,
                    'is_padded': False,
                    'original_length': end_idx - i,
                    'mel_frames': chunk_mel_frames,
                    'chunk_idx': len(chunks)
                })
        
        # Skip files that couldn't be chunked properly
        if not chunks:
            return None
        
        # Return preprocessed data with chunks
        return AudioData(
            metadata=file_metadata,
            audio=None,  # Don't store the full audio anymore
            sr=sr,
            phones=phones,
            phone_indices=phone_indices,
            start_times=start_times,
            end_times=end_times,
            durations=durations,
            audio_length=audio_length,
            audio_duration_sec=audio_duration_sec,
            phone_counts=phone_counts,
            chunks=chunks  # New field to store chunks
        )
            
    except Exception as e:
        logger.error(f"Error processing {file_metadata.lab_file}: {str(e)}\n{traceback.format_exc()}")
        return None

# Wrapper function for multiprocessing
def process_file_for_mp(args):
    """Wrapper function that can be pickled for multiprocessing."""
    file_metadata, phone_map, sample_rate, max_audio_length, max_mel_frames, hop_length = args
    return stage1_process_file(file_metadata, phone_map, sample_rate, max_audio_length, max_mel_frames, hop_length)

# Stage 2: GPU-based feature extraction
def stage2_extract_features_batch(batch_data, hop_length, win_length, n_mels, fmin, fmax, device, n_harmonics=10):
    """
    Process a batch of audio chunks on GPU for feature extraction.
    This runs in a single process to maximize GPU utilization.
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
        chunks = audio_data.chunks
        
        processed_chunks = []
        
        for chunk in chunks:
            audio = chunk['audio']
            phone_indices = chunk['phone_indices']
            start_times = chunk['start_times']
            end_times = chunk['end_times']
            mel_frames = chunk['mel_frames']
            
            # Convert audio to torch tensor
            audio_tensor = torch.FloatTensor(audio).to(device)
            
            # Extract F0 using Parselmouth (much faster than librosa.pyin)
            # Run on CPU as it's not GPU accelerated
            f0 = extract_f0_parselmouth(audio, SAMPLE_RATE, hop_length)
            f0_tensor = torch.FloatTensor(f0).to(device)
            
            # Extract harmonic amplitudes based on F0
            harmonic_amplitudes = extract_harmonic_amplitudes(
                audio,
                f0,
                SAMPLE_RATE, 
                hop_length, 
                n_harmonics,
                win_length
            )
            
            # Extract mel spectrogram
            if len(audio_tensor.shape) == 1:
                audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension
                
            # Extract mel spectrogram
            mel_spec = mel_transform(audio_tensor)
            mel_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
            mel_norm = normalize_mel(mel_db)
            
            # Move mel to CPU and convert to numpy
            mel_np = mel_norm.squeeze(0).cpu().numpy().T
            
            # Create phone sequence aligned with audio samples
            phone_seq = np.zeros(len(audio))
            for p, start, end in zip(phone_indices, start_times, end_times):
                phone_seq[start:end] = p
            
            # Create phone sequence aligned with mel frames (downsampling)
            phone_seq_mel = np.zeros(mel_frames)
            for i in range(mel_frames):
                start_idx = i * hop_length
                end_idx = min(start_idx + hop_length, len(audio))
                if start_idx < len(audio):
                    # Use majority vote to determine phone for this frame
                    unique_phones, counts = np.unique(phone_seq[start_idx:end_idx], return_counts=True)
                    if len(unique_phones) > 0:  # Make sure there's at least one phone
                        phone_seq_mel[i] = unique_phones[np.argmax(counts)]
            
            # Create F0 aligned with audio frames (upsampling)
            t_mel = np.arange(mel_frames) * hop_length / SAMPLE_RATE  # Time in seconds for each mel frame
            t_audio = np.arange(len(audio)) / SAMPLE_RATE      # Time in seconds for each audio sample
            if len(f0) > 0:
                # Make sure f0 is the right length for mel frames
                f0_padded = np.zeros(mel_frames)
                f0_len = min(len(f0), mel_frames)
                f0_padded[:f0_len] = f0[:f0_len]
                
                # Interpolate to get f0 for each audio sample
                f0_audio = np.interp(t_audio, t_mel, f0_padded)
            else:
                f0_audio = np.zeros(len(audio))
                f0_padded = np.zeros(mel_frames)
            
            # Ensure harmonic amplitudes match mel_frames
            amplitudes_padded = np.zeros((mel_frames, n_harmonics))
            amplitudes_len = min(len(harmonic_amplitudes), mel_frames)
            amplitudes_padded[:amplitudes_len] = harmonic_amplitudes[:amplitudes_len]
            
            processed_chunks.append({
                'audio': audio,
                'phone_seq': phone_seq,
                'phone_seq_mel': phone_seq_mel,
                'f0': f0_padded,
                'f0_audio': f0_audio,
                'mel': mel_np,
                'amplitudes': amplitudes_padded,
                'singer_id': metadata.singer_idx,
                'language_id': metadata.language_idx,
                'filename': f"{metadata.singer_id}_{metadata.language_id}_{metadata.base_name}_{chunk['chunk_idx']}",
                'is_padded': chunk['is_padded'],
                'original_length': chunk['original_length']
            })
        
        # Return processed data
        results.append(ProcessedFeatures(
            metadata=metadata,
            segments=processed_chunks,
            phone_counts=audio_data.phone_counts,
            audio_duration_sec=audio_data.audio_duration_sec
        ))
        
    return results

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
                 rebuild_cache=False, max_files=None,
                 n_mels=N_MELS, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, fmin=FMIN, fmax=FMAX,
                 num_workers=8, batch_size=32, device='cuda' if torch.cuda.is_available() else 'cpu',
                 n_harmonics=10, 
                 is_train=True, train_files=None, val_files=None, seed=42):
        """
        Initialize the SingingVoiceDataset.
        
        New parameters:
        - is_train: Whether this is a training dataset (True) or validation dataset (False)
        - train_files: Number of files to use for training (if None, use all available except validation files)
        - val_files: Number of files to use for validation (if None, use all available except training files)
        - seed: Random seed for reproducible file selection
        """
        self.dataset_dir = dataset_dir
        self.cache_dir = cache_dir
        self.sample_rate = sample_rate
        self.max_files = max_files
        self.n_harmonics = n_harmonics
        self.is_train = is_train
        self.train_files = train_files
        self.val_files = val_files
        self.seed = seed
        self.hop_length = hop_length
        self.win_length = win_length

        # Parameters for mel spectrogram extraction
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        
        # Multiprocessing parameters
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.device = device

        os.makedirs(cache_dir, exist_ok=True)
        
        # Create separate cache files for training and validation
        dataset_type = "train" if self.is_train else "val"
        cache_path = os.path.join(
            cache_dir, 
            f"singing_voice_{dataset_type}_data_{sample_rate}hz_{n_mels}mels.pkl"
        )
        
        logger.info(f"Initializing {'training' if is_train else 'validation'} dataset")
        
        if os.path.exists(cache_path) and not rebuild_cache:
            self.load_cache(cache_path)
        else:
            self.build_dataset_pipeline()
            self.save_cache(cache_path)
            self.generate_distribution_log(dataset_type)
    
    def build_dataset_pipeline(self):
        """Build dataset using multi-stage pipeline approach."""
        logger.info(f"Building {'training' if self.is_train else 'validation'} dataset using multi-stage pipeline...")
        
        # Stage 0: Directory scanning and max length estimation
        self.scan_dataset_structure()
        
        # Create file processing tasks
        all_tasks = self.create_processing_tasks()
        
        # Estimate max audio length and mel frames
        self.max_audio_length, self.max_mel_frames = estimate_max_lengths(
            all_tasks, self.sample_rate, self.hop_length, max_files=100
        )
        
        # Set random seed for reproducible file selection
        random.seed(self.seed)
        
        # Shuffle all tasks
        random.shuffle(all_tasks)
        
        # Select files for training and validation based on counts
        total_files = len(all_tasks)
        
        if self.is_train:
            # For training dataset
            if self.train_files is not None:
                # Use specified number of training files
                file_count = min(self.train_files, total_files)
                if self.val_files is not None:
                    # Ensure we don't overlap with validation files
                    file_count = min(file_count, total_files - self.val_files)
                
                processing_tasks = all_tasks[:file_count]
                logger.info(f"Selected {len(processing_tasks)} files for training out of {total_files} total files")
            else:
                # Use all available files for training (except validation files if specified)
                if self.val_files is not None:
                    val_count = min(self.val_files, total_files)
                    processing_tasks = all_tasks[val_count:]
                    logger.info(f"Using {len(processing_tasks)} files for training (reserved {val_count} for validation)")
                else:
                    processing_tasks = all_tasks
                    logger.info(f"Using all {len(processing_tasks)} files for training")
        else:
            # For validation dataset
            if self.val_files is not None:
                # Use specified number of validation files
                file_count = min(self.val_files, total_files)
                processing_tasks = all_tasks[:file_count]
                logger.info(f"Selected {len(processing_tasks)} files for validation out of {total_files} total files")
            else:
                # Default validation behavior - use a small portion if train_files is specified
                if self.train_files is not None:
                    train_count = min(self.train_files, total_files)
                    processing_tasks = all_tasks[train_count:]
                    logger.info(f"Using {len(processing_tasks)} files for validation (reserved {train_count} for training)")
                else:
                    # Use all files for validation if nothing is specified
                    processing_tasks = all_tasks
                    logger.info(f"Using all {len(processing_tasks)} files for validation")
        
        # Apply max_files limit if specified
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
        mappings = scan_directory_structure(self.dataset_dir)
        self.singer_map = mappings['singer_map']
        self.inv_singer_map = mappings['inv_singer_map']
        self.language_map = mappings['language_map']
        self.inv_language_map = mappings['inv_language_map']
        self.phone_map = mappings['phone_map']
        self.inv_phone_map = mappings['inv_phone_map']
    
    def create_processing_tasks(self):
        """Create list of file processing tasks."""
        return create_file_tasks(self.dataset_dir, self.singer_map, self.language_map)
    
    def run_stage1_preprocessing(self, tasks):
        """Run Stage 1: Multiprocessing for file loading and initial processing."""
        # Process files in parallel
        max_workers = self.num_workers if self.num_workers > 0 else min(32, os.cpu_count() + 4)
        
        logger.info(f"Stage 1: Processing {len(tasks)} files with {max_workers} workers")
        
        # Prepare args for multiprocessing - must be picklable
        mp_args = [(task, self.phone_map, self.sample_rate, self.max_audio_length, self.max_mel_frames, self.hop_length) for task in tasks]
        
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
        """Save dataset to cache file."""
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
            'phone_language_stats': self.phone_language_stats,
            'is_train': self.is_train,
            'max_audio_length': self.max_audio_length,
            'max_mel_frames': self.max_mel_frames
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        logger.info(f"Dataset cached to {cache_path}")
    
    def load_cache(self, cache_path):
        """Load dataset from cache file."""
        logger.info(f"Loading {'training' if self.is_train else 'validation'} dataset from cache: {cache_path}")
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
        
        # Load max dimensions
        self.max_audio_length = cache_data.get('max_audio_length')
        self.max_mel_frames = cache_data.get('max_mel_frames')
        
        logger.info(f"Loaded {len(self.data)} segments with {len(self.singer_map)} singers, "
                  f"{len(self.language_map)} languages, and {len(self.phone_map)} unique phones")
        logger.info(f"Max audio length: {self.max_audio_length} samples, Max mel frames: {self.max_mel_frames}")
    
    def generate_distribution_log(self, dataset_type="dataset"):
        """Generate a log file with dataset distribution statistics."""
        log_path = os.path.join(self.cache_dir, f"{dataset_type}_distribution.txt")
        
        with open(log_path, 'w') as f:
            f.write(f"=== SINGING VOICE {dataset_type.upper()} DISTRIBUTION ===\n\n")
            
            # Overall statistics
            f.write(f"Total segments: {len(self.data)}\n")
            f.write(f"Total singers: {len(self.singer_map)}\n")
            f.write(f"Total languages: {len(self.language_map)}\n")
            f.write(f"Total unique phones: {len(self.phone_map)}\n\n")
            f.write(f"Max audio length: {self.max_audio_length} samples ({self.max_audio_length/self.sample_rate:.2f} seconds)\n")
            f.write(f"Max mel frames: {self.max_mel_frames}\n\n")
            
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
        
        # Extract tensors - now they're all standardized sizes with padding applied during preprocessing
        audio = torch.FloatTensor(item['audio'])
        phone_seq = torch.LongTensor(item['phone_seq'])
        phone_seq_mel = torch.LongTensor(item['phone_seq_mel'])
        f0 = torch.FloatTensor(item['f0'])
        f0_audio = torch.FloatTensor(item['f0_audio'])
        mel = torch.FloatTensor(item['mel'])
        singer_id = torch.LongTensor([item['singer_id']])
        language_id = torch.LongTensor([item['language_id']])
        amplitudes = torch.FloatTensor(item['amplitudes'])
        
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
                  train_files=None, val_files=None, device='cuda', collate_fn=None, 
                  n_harmonics=10, seed=42, create_val=True):
    """
    Get dataloaders for the singing voice dataset.
    
    Args:
        batch_size: Number of samples per batch
        num_workers: Number of workers for the DataLoader
        pin_memory: Whether to pin memory in DataLoader
        persistent_workers: Whether to keep workers alive between epochs
        train_files: Number of files to use for training (if None, use all except validation)
        val_files: Number of files to use for validation (if None, use all except training)
        device: Device to use for GPU acceleration ('cuda' or 'cpu')
        collate_fn: Custom collate function (if None, use the standardized one)
        n_harmonics: Number of harmonics to extract
        seed: Random seed for reproducible file selection
        create_val: Whether to create a validation dataloader
        
    Returns:
        If create_val=True: (train_loader, val_loader, train_dataset, val_dataset)
        If create_val=False: (train_loader, train_dataset)
    """
    # Use the standardized collate function by default
    if collate_fn is None:
        collate_fn = standardized_collate_fn
    
    logger.info("Creating training dataset...")
    train_dataset = SingingVoiceDataset(
        rebuild_cache=False,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        fmin=FMIN,
        fmax=FMAX,
        num_workers=num_workers,
        device=device,
        n_harmonics=n_harmonics,
        is_train=True,
        train_files=train_files,
        val_files=val_files,
        seed=seed
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        collate_fn=collate_fn
    )
    
    if create_val:
        logger.info("Creating validation dataset...")
        val_dataset = SingingVoiceDataset(
            rebuild_cache=False,
            n_mels=N_MELS,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            fmin=FMIN,
            fmax=FMAX,
            num_workers=num_workers,
            device=device,
            n_harmonics=n_harmonics,
            is_train=False,
            train_files=train_files,
            val_files=val_files,
            seed=seed
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            collate_fn=collate_fn
        )
        
        return train_loader, val_loader, train_dataset, val_dataset
    else:
        return train_loader, train_dataset

if __name__ == "__main__":
    # Example usage
    from utils import standardized_collate_fn
    
    train_loader, val_loader, train_dataset, val_dataset = get_dataloader(
        batch_size=16, 
        num_workers=4,
        train_files=90,
        val_files=10,
        collate_fn=standardized_collate_fn
    )
    
    # Print dataset stats
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Max audio length: {train_dataset.max_audio_length} samples ({train_dataset.max_audio_length/SAMPLE_RATE:.2f} seconds)")
    print(f"Max mel frames: {train_dataset.max_mel_frames}")
    
    # Benchmark loading time
    start_time = time.time()
    for i, batch in enumerate(tqdm(train_loader, desc="Loading training batches")):
        if i > 5:  # Load a few batches for benchmark
            break
    print(f"Time to load 5 training batches: {time.time() - start_time:.2f} seconds")
    
    start_time = time.time()
    for i, batch in enumerate(tqdm(val_loader, desc="Loading validation batches")):
        if i > 5:  # Load a few batches for benchmark
            break
    print(f"Time to load 5 validation batches: {time.time() - start_time:.2f} seconds")