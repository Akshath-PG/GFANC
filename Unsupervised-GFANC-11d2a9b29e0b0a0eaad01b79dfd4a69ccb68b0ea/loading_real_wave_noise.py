import os 
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import scipy.signal as signal
import soundfile

#--------------------------------------------------------------
# untion: print_stats()
# Description : print the information of wave file.
#--------------------------------------------------------------
def print_stats(waveform, sample_rate=None, src=None):
    if src:
        print("-" * 10)
        print("Source:", src)
        print("-" * 10)
    if sample_rate:
        print("Sample Rate:", sample_rate)
        print("Shape:", tuple(waveform.shape))
        print("Dtype:", waveform.dtype)
        print(f" - Max:     {waveform.max().item():6.3f}")
        print(f" - Min:     {waveform.min().item():6.3f}")
        print(f" - Mean:    {waveform.mean().item():6.3f}")
        print(f" - Std Dev: {waveform.std().item():6.3f}")
        print()
        print(waveform)
        print()

#-------------------------------------------------------------- 
# Function: plot_waveform()
# Decription : plot the waveform of the wave file 
#--------------------------------------------------------------
def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c+1}')
        if xlim:
            axes[c].set_xlim(xlim)
        if ylim:
            axes[c].set_ylim(ylim)
    figure.suptitle(title)
    plt.show(block=True)

#--------------------------------------------------------------
# Function: plot_specgram()
# Decription : plot the powerspecgram of the wave file 
#-------------------------------------------------------------
def plot_specgram(waveform, sample_rate, title="Spectrogram", xlim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c+1}')
        if xlim:
            axes[c].set_xlim(xlim)
    figure.suptitle(title)
    plt.show(block=True)

#--------------------------------------------------------------
# Function: resample_wav()
# Description: resample the waveform using resample_rate
#--------------------------------------------------------------
def resample_wav(waveform, sample_rate, resample_rate):
    resampler = T.Resample(sample_rate, resample_rate, dtype=waveform.dtype)
    resampled_waveform = resampler(waveform)
    return resampled_waveform

#--------------------------------------------------------------
# Function: loading_real_wave_noise()
# Description: loading real noise, filter and resample it. 
#--------------------------------------------------------------
def loading_real_wave_noise(folde_name, sound_name, lowcut=100.0, highcut=500.0):
    """
    Loads audio, resamples to 16kHz, and applies a Pulmonary Stethoscope Filter (100-500Hz)
    using a 4th-order Butterworth Bandpass filter with real-time state maintenance.
    """
    # Load with soundfile to avoid torchaudio backend issues
    data, sample_rate = soundfile.read(os.path.join(folde_name, sound_name))
    waveform = torch.from_numpy(data).float()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0) # [1, samples]
    else:
        waveform = waveform.t() # [channels, samples]
        
    resample_rate = 16000
    waveform = resample_wav(waveform, sample_rate, resample_rate) # resample

    # Ensure waveform is 1D for processing (take first channel if multi-channel)
    if waveform.shape[0] > 1:
        waveform_np = waveform[0].numpy()
    else:
        waveform_np = waveform.squeeze().numpy()

    # Pulmonary Stethoscope Filter (100-500 Hz)
    if lowcut is not None and highcut is not None:
        nyquist = 0.5 * resample_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        
        # 4th-order Butterworth Bandpass Filter
        b, a = signal.butter(4, [low, high], btype='bandpass')
        
        # Initialize filter state (zi) for continuity between chunks
        # Start with zero state (silence)
        zi = signal.lfilter_zi(b, a) * 0
        
        # Real-time processing simulation (chunking)
        chunk_size = 1024
        processed_chunks = []
        
        for i in range(0, len(waveform_np), chunk_size):
            chunk = waveform_np[i:i+chunk_size]
            
            # Apply filter with state updates
            filtered_chunk, zi = signal.lfilter(b, a, chunk, zi=zi)
            processed_chunks.append(filtered_chunk)
            
        filtered_waveform = np.concatenate(processed_chunks)

        # Scaling/Normalization for visualization
        max_val = np.max(np.abs(filtered_waveform))
        if max_val > 0:
             filtered_waveform = filtered_waveform / max_val
        
        # Convert back to tensor [1, samples]
        waveform = torch.from_numpy(filtered_waveform).float().unsqueeze(0)

    return waveform, resample_rate

#----------------------------------------------------------------
# loading the wave file 
def main():
    folde_name = 'Real_noise'
    sound_name = 'Aircraft.wav'
    SAMPLE_WAV_SPEECH_PATH = os.path.join(folde_name, sound_name)
    waveform, sample_rate = torchaudio.load(SAMPLE_WAV_SPEECH_PATH)
    waveform = resample_wav(waveform, sample_rate, 16000)
    sample_rate = 16000
    print_stats(waveform, sample_rate=sample_rate)
    plot_waveform(waveform, sample_rate)
    plot_specgram(waveform, sample_rate)

if __name__ == "__main__":
    main()