import os
import torch
import numpy as np
import librosa
import librosa.feature
from torch.utils.data import Dataset, DataLoader
import pickle
import glob
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import defaultdict
import concurrent.futures
import traceback
import logging

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
HOP_LENGTH = 240  # Fixed inconsistency
WIN_LENGTH = 1024
N_MELS = 80
FMIN = 40
FMAX = 16000
MIN_PHONE = 5  # Minimum number of phones required per file
MIN_DURATION_MS = 10  # Minimum duration for a phone in milliseconds
ENABLE_ALIGNMENT_PLOTS = False  # Flag to enable/disable alignment plots

# This function processes a single audio/label file pair
# Must be at module level to be picklable for multiprocessing
def process_audio_file(args):
    """Process a single audio/lab file pair and extract features."""
    (lab_file, singer_id, language_id, singer_idx, language_idx, phone_map, 
     sample_rate, context_window_samples, hop_length, win_length, 
     n_mels, fmin, fmax, enable_alignment_plots) = args
    
    # Normalize path and get components
    lab_file = os.path.normpath(lab_file)
    lab_dir = os.path.dirname(lab_file)
    base_name = os.path.basename(lab_file).replace('.lab', '')
    
    # Go up one level (from 'lab' directory to language directory)
    lang_dir = os.path.dirname(lab_dir)
    
    # Construct wav directory path - side by side with 'lab' folder
    wav_dir = os.path.join(lang_dir, "wav")
    
    # Full path to the wav file
    wav_file = os.path.join(wav_dir, f"{base_name}.wav")
    
    # Debug info
    print(f"Lab file: {lab_file}")
    print(f"Looking for wav file at: {wav_file}")
    print(f"Wav directory exists: {os.path.exists(wav_dir)}")
    
    if not os.path.exists(wav_file):
        return {
            'success': False,
            'error': f"No matching wav file for {lab_file}. Tried {wav_file}",
            'results': []
        }
    
    # Read phone labels
    phones = []
    start_times = []
    end_times = []
    
    try:
        with open(lab_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    start, end, phone = parts
                    start_times.append(int(start))
                    end_times.append(int(end))
                    phones.append(phone)
        
        # Skip files with fewer phones than MIN_PHONE
        if len(phones) < MIN_PHONE:
            return {
                'success': True,
                'results': [],
                'singer_language_stats': {singer_id: {language_id: 0}},
                'singer_duration': {singer_id: 0},
                'language_duration': {language_id: 0},
                'phone_language_stats': {language_id: {}}
            }
        
        # Phone statistics
        phone_language_stats = defaultdict(int)
        for phone in phones:
            phone_language_stats[phone] += 1
        
        # Calculate durations
        durations = [end - start for start, end in zip(start_times, end_times)]
        
        # Adjust durations if they are too short
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
        
        audio, sr = librosa.load(wav_file, sr=sample_rate)
        audio_length = len(audio)
        
        # Update statistics
        audio_duration_sec = audio_length / sample_rate
        
        file_results = []
        
        # Helper function to extract mel spectrogram (needed inside the worker)
        def extract_mel_spectrogram(audio):
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=sample_rate,
                n_mels=n_mels,
                hop_length=hop_length,
                win_length=win_length,
                fmin=fmin,
                fmax=fmax
            )
            mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min())
            return mel_spec.T
        
        if len(start_times) > 0:
            max_time = max(end_times)
            
            # Scale the timestamps to match the audio length
            start_times = [int(t * audio_length / max_time) for t in start_times]
            end_times = [int(t * audio_length / max_time) for t in end_times]
            
            # Test and plot the alignment for debugging if enabled
            if enable_alignment_plots:
                # Simplified plotting functionality
                # Full implementation would require plot_alignment function to be picklable
                pass
            
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio, 
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=sample_rate,
                hop_length=hop_length
            )
            f0 = np.nan_to_num(f0)
            
            phone_indices = [phone_map[p] for p in phones]
            
            # Calculate the exact expected F0 length for consistency
            target_f0_length = context_window_samples // hop_length + 1
            
            if audio_length < context_window_samples:
                padded_audio = np.pad(audio, (0, context_window_samples - audio_length))
                
                # Extract mel spectrogram for padded audio
                mel_spec = extract_mel_spectrogram(padded_audio)
                
                phone_seq = np.zeros(context_window_samples)
                for p, start, end in zip(phone_indices, start_times, end_times):
                    phone_seq[start:end] = p
                
                # Ensure consistent F0 length
                f0_padded = np.zeros(target_f0_length)
                f0_len = min(len(f0), target_f0_length)
                f0_padded[:f0_len] = f0[:f0_len]
                
                file_results.append({
                    'audio': padded_audio,
                    'phone_seq': phone_seq,
                    'f0': f0_padded,
                    'mel': mel_spec,
                    'singer_id': singer_idx,
                    'language_id': language_idx,
                    'filename': f"{singer_id}_{language_id}_{base_name}"
                })
            else:
                for i in range(0, audio_length, context_window_samples):
                    if i + context_window_samples > audio_length:
                        break
                    
                    segment_audio = audio[i:i+context_window_samples]
                    
                    # Extract mel spectrogram for this segment
                    segment_mel = extract_mel_spectrogram(segment_audio)
                    
                    # Ensure consistent F0 length for each segment
                    f0_start_idx = i // hop_length
                    f0_end_idx = f0_start_idx + target_f0_length
                    
                    segment_f0 = np.zeros(target_f0_length)
                    if f0_start_idx < len(f0):
                        actual_f0_len = min(len(f0) - f0_start_idx, target_f0_length)
                        segment_f0[:actual_f0_len] = f0[f0_start_idx:f0_start_idx + actual_f0_len]
                    
                    phone_seq = np.zeros(context_window_samples)
                    for p, start, end in zip(phone_indices, start_times, end_times):
                        seg_start = max(0, start - i)
                        seg_end = min(context_window_samples, end - i)
                        if seg_end > seg_start and seg_start < context_window_samples:
                            phone_seq[seg_start:seg_end] = p
                    
                    file_results.append({
                        'audio': segment_audio,
                        'phone_seq': phone_seq,
                        'f0': segment_f0,
                        'mel': segment_mel,
                        'singer_id': singer_idx,
                        'language_id': language_idx,
                        'filename': f"{singer_id}_{language_id}_{base_name}_{i}"
                    })
        
        return {
            'success': True,
            'results': file_results,
            'singer_language_stats': {singer_id: {language_id: 1}},
            'singer_duration': {singer_id: audio_duration_sec},
            'language_duration': {language_id: audio_duration_sec},
            'phone_language_stats': {language_id: phone_language_stats}
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': f"Error processing {lab_file}: {str(e)}\n{traceback.format_exc()}",
            'results': []
        }

class SingingVoiceDataset(Dataset):
    def __init__(self, dataset_dir=DATASET_DIR, cache_dir=CACHE_DIR, sample_rate=SAMPLE_RATE,
                 context_window_samples=CONTEXT_WINDOW_SAMPLES, rebuild_cache=False, max_files=None,
                 n_mels=N_MELS, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, fmin=FMIN, fmax=FMAX,
                 num_workers=8):
        self.dataset_dir = dataset_dir
        self.cache_dir = cache_dir
        self.sample_rate = sample_rate
        self.context_window_samples = context_window_samples
        self.max_files = max_files
        
        # Parameters for mel spectrogram extraction
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.win_length = win_length
        self.fmin = fmin
        self.fmax = fmax
        
        # Number of parallel workers (None means auto-determine)
        self.num_workers = num_workers
        
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"singing_voice_multi_singer_data_{sample_rate}hz_{n_mels}mels.pkl")
        
        if os.path.exists(cache_path) and not rebuild_cache:
            self.load_cache(cache_path)
        else:
            self.build_dataset()
            self.save_cache(cache_path)
            self.generate_distribution_log()
    
    def extract_mel_spectrogram(self, audio):
        """Extract mel spectrogram from audio with specified parameters."""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            win_length=self.win_length,
            fmin=self.fmin,
            fmax=self.fmax
        )
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min())
        return mel_spec.T
    
    def build_dataset(self):
        logger.info("Building dataset...")
        
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
        logger.info(f"Singer mapping created: {self.singer_map}")
        
        # Find all language directories and create language mapping
        language_dirs = []
        for singer_dir in singer_dirs:
            lang_dirs = [d for d in glob.glob(os.path.join(singer_dir, "*")) if os.path.isdir(d)]
            logger.info(f"Singer {os.path.basename(singer_dir)} has languages: {[os.path.basename(d) for d in lang_dirs]}")
            language_dirs.extend(lang_dirs)
        
        unique_languages = set(os.path.basename(l) for l in language_dirs)
        self.language_map = {lang: i for i, lang in enumerate(sorted(unique_languages))}
        self.inv_language_map = {i: lang for lang, i in self.language_map.items()}
        logger.info(f"Language mapping created: {self.language_map}")
        
        # Find all phonemes across all languages and prepare processing tasks
        all_phones = set()
        processing_tasks = []
        total_lab_files = 0
        missing_wav_dirs = 0
        
        # First pass: scan directory structure and collect phonemes
        for singer_dir in singer_dirs:
            singer_id = os.path.basename(singer_dir)
            singer_idx = self.singer_map[singer_id]
            
            for lang_dir in glob.glob(os.path.join(singer_dir, "*")):
                if not os.path.isdir(lang_dir):
                    continue
                    
                language_id = os.path.basename(lang_dir)
                language_idx = self.language_map[language_id]
                
                # Check for lab directory
                lab_dir = os.path.join(lang_dir, "lab")
                if not os.path.exists(lab_dir):
                    logger.warning(f"No lab directory for {singer_id}/{language_id}")
                    continue
                
                # Check for wav directory
                wav_dir = os.path.join(lang_dir, "wav")
                if not os.path.exists(wav_dir):
                    logger.error(f"No wav directory for {singer_id}/{language_id}")
                    missing_wav_dirs += 1
                    continue
                
                # List lab files
                lab_files = glob.glob(os.path.join(lab_dir, "*.lab"))
                total_lab_files += len(lab_files)
                
                if not lab_files:
                    logger.warning(f"No lab files found for {singer_id}/{language_id}")
                    continue
                    
                logger.info(f"Found {len(lab_files)} lab files for {singer_id}/{language_id}")
                
                # Check for wav files
                wav_files = glob.glob(os.path.join(wav_dir, "*.wav"))
                if not wav_files:
                    logger.error(f"No wav files found for {singer_id}/{language_id}")
                    continue
                    
                logger.info(f"Found {len(wav_files)} wav files for {singer_id}/{language_id}")
                
                # First pass: collect all phones for mapping and validate file pairs
                missing_wav_count = 0
                for lab_file in lab_files:
                    base_name = os.path.basename(lab_file).replace('.lab', '')
                    expected_wav_file = os.path.join(wav_dir, f"{base_name}.wav")
                    
                    if not os.path.exists(expected_wav_file):
                        logger.warning(f"Missing wav file: {expected_wav_file} for lab: {lab_file}")
                        missing_wav_count += 1
                        continue
                    
                    # Read phones from lab file
                    try:
                        with open(lab_file, 'r') as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) == 3:
                                    _, _, phone = parts
                                    all_phones.add(phone)
                        
                        # Prepare the task for later processing
                        processing_tasks.append((
                            lab_file, singer_id, language_id, singer_idx, language_idx
                        ))
                    except Exception as e:
                        logger.error(f"Error reading lab file {lab_file}: {str(e)}")
                
                if missing_wav_count > 0:
                    logger.warning(f"Missing {missing_wav_count} wav files for {singer_id}/{language_id}")
        
        if missing_wav_dirs > 0:
            logger.error(f"Missing {missing_wav_dirs} wav directories. Check directory structure.")
            
        # Create phone mapping
        self.phone_map = {phone: i for i, phone in enumerate(sorted(all_phones))}
        self.inv_phone_map = {i: phone for phone, i in self.phone_map.items()}
        logger.info(f"Found {len(self.phone_map)} unique phones across {total_lab_files} lab files")
        logger.info(f"Prepared {len(processing_tasks)} valid file pairs for processing")
        
        # Limit the number of files if max_files is specified
        if self.max_files and self.max_files < len(processing_tasks):
            print(f"Limiting dataset to {self.max_files} files out of {len(processing_tasks)}")
            processing_tasks = processing_tasks[:self.max_files]
        
        # Initialize statistics counters
        self.singer_language_stats = defaultdict(lambda: defaultdict(int))
        self.singer_duration = defaultdict(float)
        self.language_duration = defaultdict(float)
        self.phone_language_stats = defaultdict(lambda: defaultdict(int))
        
        # Process files in parallel
        self.data = []
        
        # Determine the number of workers
        if self.num_workers is None:
            max_workers = min(32, os.cpu_count() + 4)  # Common formula for CPU-bound tasks
        else:
            max_workers = self.num_workers
        
        print(f"Processing files with {max_workers} workers")
        
        # Prepare args for each task (adding all necessary parameters)
        processing_args = []
        for lab_file, singer_id, language_id, singer_idx, language_idx in processing_tasks:
            task_args = (
                lab_file, singer_id, language_id, singer_idx, language_idx, 
                self.phone_map, self.sample_rate, self.context_window_samples, 
                self.hop_length, self.win_length, self.n_mels, self.fmin, 
                self.fmax, ENABLE_ALIGNMENT_PLOTS
            )
            processing_args.append(task_args)
        
        # Use ProcessPoolExecutor for parallel processing
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Process files in parallel and track progress with tqdm
            results = list(tqdm(
                executor.map(process_audio_file, processing_args),
                total=len(processing_args),
                desc="Processing files"
            ))
        
        # Process results
        for result in tqdm(results, desc="Collecting results"):
            if not result['success']:
                print(f"Warning: {result['error']}")
                continue
            
            # Extend dataset with file results
            self.data.extend(result['results'])
            
            # Update statistics
            for singer_id, languages in result['singer_language_stats'].items():
                for language_id, count in languages.items():
                    self.singer_language_stats[singer_id][language_id] += count
            
            for singer_id, duration in result['singer_duration'].items():
                self.singer_duration[singer_id] += duration
            
            for language_id, duration in result['language_duration'].items():
                self.language_duration[language_id] += duration
            
            for language_id, phones in result['phone_language_stats'].items():
                for phone, count in phones.items():
                    self.phone_language_stats[language_id][phone] += count
        
        print(f"Dataset built with {len(self.data)} segments, {len(self.singer_map)} singers, {len(self.language_map)} languages, and {len(self.phone_map)} unique phones")
    
    def plot_alignment(self, audio, phones, start_times, end_times, base_name):
        """Plot the alignment between audio waveform and phone sequence for verification."""
        alignments_dir = os.path.join(self.cache_dir, "alignments")
        os.makedirs(alignments_dir, exist_ok=True)
        
        plt.figure(figsize=(15, 8))
        
        # Plot audio waveform
        plt.subplot(2, 1, 1)
        time = np.arange(len(audio)) / self.sample_rate
        plt.plot(time, audio)
        plt.title(f'Audio Waveform - {base_name}')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        
        # Plot phone segments
        plt.subplot(2, 1, 2)
        
        # Convert sample positions to seconds
        start_times_sec = [s / self.sample_rate for s in start_times]
        end_times_sec = [e / self.sample_rate for e in end_times]
        
        # Plot phone labels and boundaries
        for i, (phone, start, end) in enumerate(zip(phones, start_times_sec, end_times_sec)):
            plt.axvline(x=start, color='r', linestyle='--', alpha=0.5)
            plt.axvline(x=end, color='r', linestyle='--', alpha=0.5)
            plt.text((start + end) / 2, 0.5, phone, 
                     horizontalalignment='center', verticalalignment='center',
                     bbox=dict(facecolor='white', alpha=0.7))
        
        plt.plot(time, np.zeros_like(time), color='black', alpha=0.3)
        plt.title('Phone Alignment')
        plt.xlabel('Time (s)')
        plt.ylim(-1, 1)
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(alignments_dir, f'{base_name}_alignment.png'))
        plt.close()
    
    def save_cache(self, cache_path):
        cache_data = {
            'data': self.data,
            'phone_map': self.phone_map,
            'inv_phone_map': self.inv_phone_map,
            'singer_map': self.singer_map,
            'inv_singer_map': self.inv_singer_map,
            'language_map': self.language_map,
            'inv_language_map': self.inv_language_map
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"Dataset cached to {cache_path}")
    
    def load_cache(self, cache_path):
        print(f"Loading dataset from cache: {cache_path}")
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        self.data = cache_data['data']
        self.phone_map = cache_data['phone_map']
        self.inv_phone_map = cache_data['inv_phone_map']
        self.singer_map = cache_data['singer_map']
        self.inv_singer_map = cache_data['inv_singer_map']
        self.language_map = cache_data['language_map']
        self.inv_language_map = cache_data['inv_language_map']
        print(f"Loaded {len(self.data)} segments with {len(self.singer_map)} singers, {len(self.language_map)} languages, and {len(self.phone_map)} unique phones")
    
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
            for singer_id, singer_idx in self.singer_map.items():
                count = sum(self.singer_language_stats[singer_id].values())
                duration = self.singer_duration[singer_id]
                f.write(f"Singer {singer_id} (ID: {singer_idx}):\n")
                f.write(f"  - Files: {count}\n")
                f.write(f"  - Total duration: {duration:.2f} seconds\n")
                f.write(f"  - Languages: {', '.join(self.singer_language_stats[singer_id].keys())}\n\n")
            
            # Language statistics
            f.write("=== LANGUAGE STATISTICS ===\n")
            for language_id, language_idx in self.language_map.items():
                singer_count = sum(1 for s in self.singer_language_stats if language_id in self.singer_language_stats[s])
                file_count = sum(self.singer_language_stats[s][language_id] for s in self.singer_language_stats if language_id in self.singer_language_stats[s])
                duration = self.language_duration[language_id]
                f.write(f"Language {language_id} (ID: {language_idx}):\n")
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
            
            # Singer-Language distribution
            f.write("=== SINGER-LANGUAGE DISTRIBUTION ===\n")
            for singer_id in self.singer_language_stats:
                for language_id, count in self.singer_language_stats[singer_id].items():
                    f.write(f"Singer {singer_id}, Language {language_id}: {count} files\n")
            
            f.write("\n=== END OF DISTRIBUTION LOG ===\n")
        
        print(f"Distribution log written to {log_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        audio = torch.FloatTensor(item['audio'])
        phone_seq = torch.LongTensor(item['phone_seq'])
        f0 = torch.FloatTensor(item['f0'])
        mel = torch.FloatTensor(item['mel'])
        singer_id = torch.LongTensor([item['singer_id']])
        language_id = torch.LongTensor([item['language_id']])
        
        # Create one-hot encodings
        phone_one_hot = F.one_hot(phone_seq.long(), num_classes=len(self.phone_map)).float()
        singer_one_hot = F.one_hot(singer_id.long(), num_classes=len(self.singer_map)).float().squeeze(0)
        language_one_hot = F.one_hot(language_id.long(), num_classes=len(self.language_map)).float().squeeze(0)
        
        return {
            'audio': audio,
            'phone_seq': phone_seq,
            'phone_one_hot': phone_one_hot,
            'f0': f0,
            'mel': mel,
            'singer_id': singer_id,
            'language_id': language_id,
            'singer_one_hot': singer_one_hot,
            'language_one_hot': language_one_hot,
            'filename': item['filename']
        }

def get_dataloader(batch_size=16, num_workers=4, pin_memory=True, persistent_workers=True, max_files=None, dataset_workers=None):
    """
    Get a dataloader for the singing voice dataset.
    
    Args:
        batch_size: Number of samples per batch
        num_workers: Number of workers for the DataLoader
        pin_memory: Whether to pin memory in DataLoader
        persistent_workers: Whether to keep workers alive between epochs
        max_files: Maximum number of files to process
        dataset_workers: Number of workers for dataset processing (None = auto)
    """
    dataset = SingingVoiceDataset(
        rebuild_cache=False, 
        max_files=max_files,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        fmin=FMIN,
        fmax=FMAX,
        num_workers=dataset_workers
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False
    )
    return dataloader, dataset