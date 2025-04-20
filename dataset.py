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

DATASET_DIR = "./datasets/gin"
CACHE_DIR = "./cache"
SAMPLE_RATE = 24000
CONTEXT_WINDOW_SEC = 2
CONTEXT_WINDOW_SAMPLES = SAMPLE_RATE * CONTEXT_WINDOW_SEC
HOP_LENGTH = 240 #512
MIN_PHONE = 5  # Minimum number of phones required per file
MIN_DURATION_MS = 10  # Minimum duration for a phone in milliseconds

class SingingVoiceDataset(Dataset):
    def __init__(self, dataset_dir=DATASET_DIR, cache_dir=CACHE_DIR, sample_rate=SAMPLE_RATE,
                 context_window_samples=CONTEXT_WINDOW_SAMPLES, rebuild_cache=False, max_files=None,
                 n_mels=80, hop_length=240, win_length=1024, fmin=40, fmax=16000):
        self.dataset_dir = dataset_dir
        self.cache_dir = cache_dir
        self.sample_rate = sample_rate
        self.context_window_samples = context_window_samples
        self.max_files = max_files  # Parameter to limit the number of files loaded
        
        # Parameters for mel spectrogram extraction
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.win_length = win_length
        self.fmin = fmin
        self.fmax = fmax
        
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"singing_voice_data_{sample_rate}hz_{n_mels}mels.pkl")
        
        if os.path.exists(cache_path) and not rebuild_cache:
            self.load_cache(cache_path)
        else:
            self.build_dataset()
            self.save_cache(cache_path)
    
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
        # Convert to log scale (dB)
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        # Normalize to [0, 1] range
        mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min())
        return mel_spec
    
    def build_dataset(self):
        print("Building dataset...")
        lab_files = glob.glob(os.path.join(self.dataset_dir, "lab", "*.lab"))
        
        # Limit the number of files if max_files is specified
        if self.max_files and self.max_files < len(lab_files):
            print(f"Limiting dataset to {self.max_files} files")
            lab_files = lab_files[:self.max_files]
        
        all_phones = set()
        for lab_file in lab_files:
            with open(lab_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 3:
                        _, _, phone = parts
                        all_phones.add(phone)
        
        self.phone_map = {phone: i for i, phone in enumerate(sorted(all_phones))}
        self.inv_phone_map = {i: phone for phone, i in self.phone_map.items()}
        
        self.data = []
        for lab_file in tqdm(lab_files):
            base_name = os.path.basename(lab_file).replace('.lab', '')
            wav_file = os.path.join(self.dataset_dir, "wav", f"{base_name}.wav")
            
            if not os.path.exists(wav_file):
                print(f"Warning: No matching wav file for {lab_file}")
                continue
            
            # Read phone labels
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
            if len(phones) < MIN_PHONE:
                print(f"Skipping {lab_file} - only has {len(phones)} phones (minimum required: {MIN_PHONE})")
                continue
            
            # Calculate durations
            durations = [end - start for start, end in zip(start_times, end_times)]
            
            # Adjust durations if they are too short - borrow from neighbors
            min_duration_samples = int(MIN_DURATION_MS * 10000)  # Convert to the same units as the timestamps
            
            for i in range(len(durations)):
                if durations[i] < min_duration_samples:
                    print(f"Phone {phones[i]} has short duration: {durations[i]/10000:.2f}ms")
                    
                    # Try to borrow from left neighbor
                    if i > 0 and durations[i-1] > min_duration_samples * 2:
                        borrow_amount = min(min_duration_samples - durations[i], durations[i-1] - min_duration_samples)
                        start_times[i] -= borrow_amount
                        end_times[i-1] -= borrow_amount
                        durations[i] += borrow_amount
                        durations[i-1] -= borrow_amount
                        print(f"  Borrowed {borrow_amount/10000:.2f}ms from left neighbor ({phones[i-1]})")
                    
                    # If still too short, try to borrow from right neighbor
                    if durations[i] < min_duration_samples and i < len(durations) - 1 and durations[i+1] > min_duration_samples * 2:
                        borrow_amount = min(min_duration_samples - durations[i], durations[i+1] - min_duration_samples)
                        end_times[i] += borrow_amount
                        start_times[i+1] += borrow_amount
                        durations[i] += borrow_amount
                        durations[i+1] -= borrow_amount
                        print(f"  Borrowed {borrow_amount/10000:.2f}ms from right neighbor ({phones[i+1]})")
                    
                    # Update the duration
                    durations[i] = end_times[i] - start_times[i]
                    print(f"  Final duration: {durations[i]/10000:.2f}ms")
            
            audio, sr = librosa.load(wav_file, sr=self.sample_rate)
            audio_length = len(audio)
            
            if len(start_times) > 0:
                max_time = max(end_times)
                
                # Scale the timestamps to match the audio length
                start_times = [int(t * audio_length / max_time) for t in start_times]
                end_times = [int(t * audio_length / max_time) for t in end_times]
                durations = [end - start for start, end in zip(start_times, end_times)]
                
                # Test and plot the alignment for debugging
                self.plot_alignment(audio, phones, start_times, end_times, base_name)
                
                f0, voiced_flag, voiced_probs = librosa.pyin(
                    audio, 
                    fmin=librosa.note_to_hz('C2'),
                    fmax=librosa.note_to_hz('C7'),
                    sr=self.sample_rate,
                    hop_length=self.hop_length
                )
                f0 = np.nan_to_num(f0)
                
                phone_indices = [self.phone_map[p] for p in phones]
                
                # Calculate the exact expected F0 length for consistency
                target_f0_length = self.context_window_samples // self.hop_length + 1
                
                if audio_length < self.context_window_samples:
                    padded_audio = np.pad(audio, (0, self.context_window_samples - audio_length))
                    
                    # Extract mel spectrogram for padded audio
                    mel_spec = self.extract_mel_spectrogram(padded_audio)
                    
                    phone_seq = np.zeros(self.context_window_samples)
                    for p, start, end in zip(phone_indices, start_times, end_times):
                        phone_seq[start:end] = p
                    
                    # Ensure consistent F0 length
                    f0_padded = np.zeros(target_f0_length)
                    f0_len = min(len(f0), target_f0_length)
                    f0_padded[:f0_len] = f0[:f0_len]
                    
                    self.data.append({
                        'audio': padded_audio,
                        'phone_seq': phone_seq,
                        'f0': f0_padded,
                        'mel': mel_spec,  # Add mel spectrogram
                        'filename': base_name
                    })
                else:
                    for i in range(0, audio_length, self.context_window_samples):
                        if i + self.context_window_samples > audio_length:
                            break
                        
                        segment_audio = audio[i:i+self.context_window_samples]
                        
                        # Extract mel spectrogram for this segment
                        segment_mel = self.extract_mel_spectrogram(segment_audio)
                        
                        # Ensure consistent F0 length for each segment
                        f0_start_idx = i // self.hop_length
                        f0_end_idx = f0_start_idx + target_f0_length
                        
                        segment_f0 = np.zeros(target_f0_length)
                        if f0_start_idx < len(f0):
                            actual_f0_len = min(len(f0) - f0_start_idx, target_f0_length)
                            segment_f0[:actual_f0_len] = f0[f0_start_idx:f0_start_idx + actual_f0_len]
                        
                        phone_seq = np.zeros(self.context_window_samples)
                        for p, start, end in zip(phone_indices, start_times, end_times):
                            seg_start = max(0, start - i)
                            seg_end = min(self.context_window_samples, end - i)
                            if seg_end > seg_start and seg_start < self.context_window_samples:
                                phone_seq[seg_start:seg_end] = p
                        
                        self.data.append({
                            'audio': segment_audio,
                            'phone_seq': phone_seq,
                            'f0': segment_f0,
                            'mel': segment_mel,  # Add mel spectrogram
                            'filename': f"{base_name}_{i}"
                        })
        
        print(f"Dataset built with {len(self.data)} segments and {len(self.phone_map)} unique phones")
    
    def plot_alignment(self, audio, phones, start_times, end_times, base_name):
        """Plot the alignment between audio waveform and phone sequence for verification."""
        # Create directory for alignments if it doesn't exist
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
        
        plt.plot(time, np.zeros_like(time), color='black', alpha=0.3)  # Zero line for reference
        plt.title('Phone Alignment')
        plt.xlabel('Time (s)')
        plt.ylim(-1, 1)
        plt.grid(True)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(os.path.join(alignments_dir, f'{base_name}_alignment.png'))
        plt.close()
    
    def save_cache(self, cache_path):
        cache_data = {
            'data': self.data,
            'phone_map': self.phone_map,
            'inv_phone_map': self.inv_phone_map
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
        print(f"Loaded {len(self.data)} segments with {len(self.phone_map)} unique phones")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        audio = torch.FloatTensor(item['audio'])
        phone_seq = torch.LongTensor(item['phone_seq'])
        f0 = torch.FloatTensor(item['f0'])
        mel = torch.FloatTensor(item['mel'])  # Convert mel spectrogram to tensor
        
        # Ensure one-hot encoding uses the correct number of classes
        phone_one_hot = F.one_hot(phone_seq.long(), num_classes=len(self.phone_map)).float()
        
        return {
            'audio': audio,
            'phone_seq': phone_seq,
            'phone_one_hot': phone_one_hot,
            'f0': f0,
            'mel': mel,  # Add mel spectrogram to returned dict
            'filename': item['filename']
        }

def get_dataloader(batch_size=16, num_workers=4, pin_memory=True, persistent_workers=True, max_files=None):
    # Initialize dataset with mel spectrogram parameters
    dataset = SingingVoiceDataset(
        rebuild_cache=False, 
        max_files=max_files,
        n_mels=128,         # As specified
        hop_length=512,     # As specified
        win_length=2048,    # As specified
        fmin=40,            # As specified
        fmax=16000          # As specified
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