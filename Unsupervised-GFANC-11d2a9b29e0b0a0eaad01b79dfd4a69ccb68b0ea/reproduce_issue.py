import os
import torch
import numpy as np
import soundfile as sf
import scipy.signal as signal
import scipy.io as sio
import matplotlib.pyplot as plt
from GFANC_components import CoProcessorCNN, GFANC_Controller

# --- Utils ---
def load_and_resample(path, target_fs=16000):
    data, fs = sf.read(path)
    if data.ndim > 1: data = data[:, 0]
    if fs != target_fs:
        num_samples = int(len(data) * target_fs / fs)
        data = signal.resample(data, num_samples)
    return torch.from_numpy(data).float(), target_fs

def load_mat_tensor(path, key_guess):
    try:
        mat = sio.loadmat(path)
        # Try to find the key
        if key_guess in mat:
            val = mat[key_guess]
        else:
            # Fallback: look for likely candidates
            keys = [k for k in mat.keys() if not k.startswith('__')]
            print(f"Key '{key_guess}' not found in {path}. Found: {keys}. Using {keys[0]}.")
            val = mat[keys[0]]
        return torch.from_numpy(val).float()
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

# --- Main ---
def main():
    # 1. Config
    fs = 16000
    block_size = 512      # 32ms latency
    cnn_context_len = 2048 # 128ms context for CNN
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Block Size: {block_size} samples ({block_size/fs*1000:.1f} ms)")
    print(f"CNN Context: {cnn_context_len} samples ({cnn_context_len/fs*1000:.1f} ms)")

    folder_real = 'Noise Examples/Real_noises'
    primary_file = 'Heart_file.wav'
    reference_file = 'Anc_mic.wav'
    
    sub_filter_path = 'models/Pretrained_Sub_Control_filters_RealPath.mat'
    sec_path_file = 'Pz and Sz/RealPath/Secondary_path.mat'

    # 2. Load Data
    print("Loading Audio...")
    d_full, _ = load_and_resample(os.path.join(folder_real, primary_file), fs)
    x_full, _ = load_and_resample(os.path.join(folder_real, reference_file), fs)
    
    # Truncate to multiple of block_size for simplicity
    min_len = min(len(d_full), len(x_full))
    num_blocks = min_len // block_size
    min_len = num_blocks * block_size
    
    d_full = d_full[:min_len]
    x_full = x_full[:min_len]
    print(f"Total samples: {min_len} ({min_len/fs:.2f}s) -> {num_blocks} blocks")

    # 3. Load System Paths
    print("Loading System Paths...")
    sub_filters = load_mat_tensor(sub_filter_path, 'Wc_v')
    if sub_filters is None: return
    print(f"Sub-filters shape: {sub_filters.shape}")
    
    sec_path = load_mat_tensor(sec_path_file, 'Sz') 
    if sec_path is None: return
    
    if sec_path.ndim == 2 and sec_path.shape[0] > sec_path.shape[1]: 
        sec_path = sec_path.t() 
    if sec_path.ndim == 1:
        sec_path = sec_path.unsqueeze(0).unsqueeze(0) 
    elif sec_path.ndim == 2:
        sec_path = sec_path.unsqueeze(0) 
        
    print(f"Secondary path shape: {sec_path.shape}")

    # 4. Initialize Models
    M_sub = sub_filters.shape[0]
    cnn = CoProcessorCNN(M=M_sub).to(device)
    controller = GFANC_Controller(sub_filters, sec_path).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(cnn.parameters(), lr=1e-3)

    # 5. Online Operation Loop
    output_signal = []
    error_signal = []
    
    # State variables for continuous filtering
    prev_x = None
    prev_y = None
    
    print("Starting Real-Time Unsupervised GFANC Simulation...")
    cnn.train()
    
    # Pad input x at the beginning for CNN context
    # format: [1, 1, full_len]
    x_full_padded = torch.cat((torch.zeros(cnn_context_len - block_size), x_full), dim=0)
    
    for i in range(num_blocks):
        start_idx = i * block_size
        end_idx = start_idx + block_size
        
        # Prepare Data Chunks
        d_chunk = d_full[start_idx:end_idx].view(1, 1, -1).to(device)
        x_chunk = x_full[start_idx:end_idx].view(1, 1, -1).to(device)
        
        # Prepare CNN Input (Sliding Window with Context)
        # We need the current block + (context - block) history
        # x_full_padded aligns such that x_full_padded[i*block_size : i*block_size + cnn_context_len] 
        # ends exactly at the current block end.
        cnn_start = i * block_size
        cnn_end = cnn_start + cnn_context_len
        cnn_input = x_full_padded[cnn_start:cnn_end].view(1, 1, -1).to(device)
        
        # --- Optimization Step ---
        optimizer.zero_grad()
        
        # Forward
        # 1. CNN Estimate Weights (using context)
        weights = cnn(cnn_input) # [1, M]
        
        # 2. Controller Application (using current block + state)
        # Returns: error chunk e, anti-noise y', W, and NEW states
        e_chunk, y_prime_chunk, _, next_prev_x, next_prev_y = controller(
            x_chunk, d_chunk, weights, prev_x, prev_y
        )
        
        # 3. Loss
        loss = torch.sum(e_chunk ** 2)
        
        # --- Safety Check & Update ---
        mse_in_val = torch.mean(d_chunk ** 2).item()
        mse_out_val = torch.mean(e_chunk ** 2).item()
        
        if mse_out_val > mse_in_val:
            # Amplification detected!
            final_chunk = d_chunk.detach().cpu().numpy().flatten()
            reduction_frame = 10 * np.log10(mse_out_val / (mse_in_val + 1e-12))
            
            if (i+1) % 100 == 0:
                print(f"Block {i+1}/{num_blocks} - AMPLIFICATION ({reduction_frame:.2f} dB). Pushed Input.")
            
            optimizer.zero_grad()
            # Note: We still update valid states (prev_x, prev_y) because time moves forward!
            # The filter state must reflect that we processed this audio, even if we discard the output weights' effect for *training*, 
            # ideally we would revert the weights? No, in real-time we outputted something. 
            # If we decide to 'revert to input' for the OUTPUT audio, we just save d_chunk.
        else:
            loss.backward()
            optimizer.step()
            final_chunk = e_chunk.detach().cpu().numpy().flatten()
            reduction_frame = 10 * np.log10(mse_out_val / (mse_in_val + 1e-12))
            
            if (i+1) % 100 == 0:
                print(f"Block {i+1}/{num_blocks} - Loss: {loss.item():.4f} | Reduction: {reduction_frame:.2f} dB")

        # Update States
        prev_x = next_prev_x.detach() # Detach state to prevent infinite backprop graph growth
        prev_y = next_prev_y.detach()
        
        # Store Result
        error_signal.append(final_chunk)

    # Concatenate results
    full_error_signal = np.concatenate(error_signal)
    
    # 6. Pulmonary Filter (Post-processing)
    print("Applying Pulmonary Bandpass Filter (100-500Hz)...")
    nyquist = 0.5 * fs
    b, a = signal.butter(4, [100/nyquist, 500/nyquist], btype='bandpass')
    
    # We use lfilter_zi for stateful filtering if we were doing this online, but here it's post-processing
    final_output = signal.lfilter(b, a, full_error_signal)
    
    # 7. Save and Evaluate
    out_path = 'gfanc_output.wav'
    sf.write(out_path, final_output, fs)
    print(f"Saved output to {out_path}")
    
    # Metrics
    d_ref = d_full.numpy()
    
    mse_in = np.mean(d_ref**2)
    mse_out = np.mean(final_output**2)
    reduction = 10 * np.log10(mse_out / (mse_in + 1e-9))
    print(f"Final MSE Input: {mse_in:.6f}, Output: {mse_out:.6f}")
    print(f"Noise Reduction: {reduction:.2f} dB") 
    
    print("Generating Plot...")
    Time = np.arange(len(d_ref)) * (1/fs)
    
    plt.figure(figsize=(10, 8))
    
    plt.subplot(3, 1, 1)
    plt.title('Original Disturbance (Input)')
    plt.plot(Time, d_ref, color='blue', alpha=0.7)
    plt.grid()
    
    plt.subplot(3, 1, 2)
    plt.title('GFANC Error Signal (Before Bandpass)')
    plt.plot(Time, full_error_signal, color='orange', alpha=0.7)
    plt.grid()
    
    plt.subplot(3, 1, 3)
    plt.title('Final Output (GFANC + Bandpass 100-500Hz)')
    plt.plot(Time, final_output, color='green', alpha=0.7)
    plt.grid()
    
    plt.tight_layout()
    plt.savefig('gfanc_plot.png')
    print("Saved gfanc_plot.png")

if __name__ == "__main__":
    main()
