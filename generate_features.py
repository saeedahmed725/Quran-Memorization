import numpy as np
import tensorflow as tf
import scipy.io.wavfile
import os

# Constants
FRAME_SIZE_S = 0.025
FRAME_STRIDE_S = 0.01
STFT_NUM_POINTS = 512
NUM_TRIANGULAR_FILTERS = 40
NUM_MFCCS = 13

def generate_mfcc(signal, sample_rate_hz, chunk_duration=10, num_mfccs=NUM_MFCCS, frame_size_s=FRAME_SIZE_S, frame_stride_s=FRAME_STRIDE_S,
                  window_fn=tf.signal.hamming_window, fft_num_points=STFT_NUM_POINTS,
                  lower_freq_hz=0.0, num_mel_bins=NUM_TRIANGULAR_FILTERS, log_offset=1e-6):

    chunk_samples = int(chunk_duration * sample_rate_hz)
    mfccs_list = []

    for i in range(0, len(signal), chunk_samples):
        chunk = signal[i:i+chunk_samples]
        
        # Convert the signal to a tf tensor
        chunk = tf.convert_to_tensor(chunk, dtype=tf.float32)

        # Compute frame parameters
        frame_length = int(sample_rate_hz * frame_size_s)
        frame_step = int(sample_rate_hz * frame_stride_s)
        
        # Package the signal into frames
        frames = tf.signal.frame(chunk, frame_length=frame_length, frame_step=frame_step, pad_end=True, pad_value=0)

        # Apply Short-Time Fourier Transform
        stfts = tf.signal.stft(frames, frame_length=frame_length, frame_step=frame_step, 
                               fft_length=fft_num_points, window_fn=window_fn)

        # Compute power spectrograms
        power_spectrograms = tf.abs(stfts)**2

        # Create mel filter banks
        num_spectrogram_bins = fft_num_points // 2 + 1
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins, num_spectrogram_bins, sample_rate_hz, lower_freq_hz, sample_rate_hz/2)

        # Apply mel filterbanks to the power spectrogram
        mel_spectrograms = tf.tensordot(power_spectrograms, linear_to_mel_weight_matrix, 1)

        # Compute log mel spectrograms
        log_mel_spectrograms = tf.math.log(mel_spectrograms + log_offset)

        # Compute MFCCs
        chunk_mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[..., :num_mfccs]
        mfccs_list.append(chunk_mfccs)

    return tf.concat(mfccs_list, axis=0)

def process_audio_files(audio_dir, output_dir):
    for surah_folder in os.listdir(audio_dir):
        surah_path = os.path.join(audio_dir, surah_folder)
        if os.path.isdir(surah_path):
            surah_num = int(surah_folder.split('_')[-1])
            
            for ayah_folder in os.listdir(surah_path):
                ayah_path = os.path.join(surah_path, ayah_folder)
                if os.path.isdir(ayah_path):
                    ayah_num = int(ayah_folder.split('_')[-1])
                    
                    for audio_file in os.listdir(ayah_path):
                        if audio_file.endswith('.wav'):
                            audio_file_path = os.path.join(ayah_path, audio_file)
                            
                            # Read audio file
                            sample_rate_hz, signal = scipy.io.wavfile.read(audio_file_path)
                            
                            # Generate MFCC
                            mfcc = generate_mfcc(signal, sample_rate_hz)
                            mfcc_np = tf.keras.backend.get_value(mfcc)  # Convert to numpy array
                            
                            # Create output directory
                            output_surah_dir = os.path.join(output_dir, f's{surah_num}')
                            output_ayah_dir = os.path.join(output_surah_dir, f'a{ayah_num}')
                            os.makedirs(output_ayah_dir, exist_ok=True)
                            
                            # Save MFCC
                            output_file = os.path.join(output_ayah_dir, f'mfcc_{audio_file[:-4]}.npy')
                            np.save(output_file, mfcc_np)
                            
                            print(f"Processed: Surah {surah_num}, Ayah {ayah_num}, File: {audio_file}")

# Set your directories
audio_dir = 'word_by_word/alafasy'
output_dir = 'word_by_word_mfcc/alafasy'

# Process all audio files
process_audio_files(audio_dir, output_dir)