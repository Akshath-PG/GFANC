# How to Get GFANC Output: A Step-by-Step Guide

This document explains **every single detail** of how the "Unsupervised-GFANC" (Generative Fixed-Filter Active Noise Control) system works. It describes the data used, the training process, the barriers applied to the audio, and the exact code logic used to generate the final noise cancellation output.

It is written for two audiences:
1.  **Complete Beginners:** People with no technical background (using simple analogies).
2.  **Technical Engineers:** People who need to know the exact tensor shapes, filter frequencies, and library functions.

---

## 1. Introduction: What is this?

### 🎧 The Simple Explanation
Imagine you have a pair of Noise Cancelling Headphones. Normally, these headphones just listen to noise and try to cancel it as it happens. 
This project is building a **"Smarter Brain"** for those headphones. Instead of just reacting, this AI "Brain" listens to a 1-second clip of noise (like a drill or an engine), instantly recognizes it, and creates a custom "Anti-Noise Recipe" to cancel it out perfectly.

### 🤖 The Technical Explanation
This is an **Unsupervised Deep Learning ANC System**. It uses a **1D Convolutional Neural Network (CNN)** to analyze raw audio waveforms. The model predicts a set of 15 weights (soft labels). These weights are used to linearly combine 15 pre-trained fixed "Sub-Filters" to generate a single optimal Control Filter ($W$) for that specific noise block. The system runs in real-time, optimizing itself via backpropagation on the error signal without needing a "clean" ground truth target (hence "unsupervised").

---

## 2. The Ingredients (Data & Resources)

To get the final output (`gfanc_output.wav`), the system used the following "Ingredients".

### A. The Noises (Audio Files)
We used two specific audio files located in the `Noise Examples/Real_noises` folder:
1.  **`Heart_file.wav` (The "Disturbance"):**
    *   **What is it?** This represents the sound the user *wants* to hear (e.g., a heartbeat through a stethoscope) mixed with noise, OR simply the noise field itself depending on the test setup. In this specific run, it acted as the **Primary Noise** source we want to cancel.
    *   **Barrier Applied:** It was restricted to frequencies between **100 Hz and 500 Hz** (see Section 4).
    
2.  **`Anc_mic.wav` (The "Reference"):**
    *   **What is it?** This is the sound picked up by the outer microphone on the headphones (Reference Microphone). It hears the noise *before* it hits your ear. The AI uses this to predict the anti-noise.

### B. The Physics Models (Acoustic Paths)
Real sound bounces off walls. We simulated this using MATLAB `.mat` files:
1.  **`Secondary_path.mat` ($S(z)$):**
    *   **What is it?** A mathematical model of how sound travels from the "CANCEL SPEAKER" to the "ERROR MIC" inside the ear cup.
    *   **Why?** The AI needs to know that if it plays a sound, it will change slightly before it hits the ear. This file contains that change.

### C. The Pre-Trained Knowledge (AI Weights)
The AI wasn't born knowing how to cancel noise. It was pre-trained using a **Synthesized Dataset**.
*   **File:** `1DCNN_SyntheticDataset_UnsupervisedLearning.pth`
*   **What is it?** A file containing millions of numbers (weights) that the Neural Network learned by listening to thousands of fake noises during a previous "Training Phase". This allows it to work instantly on new real noises.

---

## 3. The Kitchen Tools (Libraries & Code)

We used Python code to "cook" this data. Here are the tools and what they did:

*   **`PyTorch` (`torch`):** The main engine. It built the Neural Network layers (`Conv1d`, `ReLU`, `Sigmoid`) and handled the math (Gradient Descent) to optimize the filter.
*   **`SciPy` (`scipy.signal`):** used to design the **Butterworth Bandpass Filter** (the frequency barrier) and to re-size audio (resampling).
*   **`NumPy`:** Used for fast math on lists of numbers (arrays).
*   **`SoundFile` (`sf`):** Used to open the `.wav` audio files and save the final result.
*   **`Matplotlib`:** Used to draw the blue/orange/green graphs at the end showing noise vs. silence.

---

## 4. The Recipe (Data Pipeline)

This is the exact step-by-step process the code followed.

### Step 1: Pre-Processing (Standardization)
*   **Action:** The code (`load_and_resample` function) loaded the `.wav` files.
*   **Detail:** It checked the **Sample Rate**. If the audio wasn't **16,000 Hz** (16,000 samples per second), it resized it.
*   **Why?** The AI model is designed to accept exactly 16,000 numbers for 1 second of audio. If we fed it 44,100 (standard music quality), it would be confused.

### Step 2: The Frequency Barrier (Crucial Detail)
*   **The Limitation:** We applied a strict **Bandpass Filter**.
*   **The Range:** **100 Hz to 500 Hz**.
*   **Why?** Active Noise Cancellation is physically very difficult for high-pitched sounds (treble). It works best for low drones (bass). We also analyzed the audio energy and found 64% of the noise was below 100Hz (thumps), which we didn't want to waste energy on. The 100-500Hz range preserved the important texture of the sound while removing deep rumbles and high hisses.
*   **Code Used:** `scipy.signal.butter(4, [100/nyquist, 500/nyquist], btype='bandpass')`

### Step 3: Looping
*   **Constraint:** The `Heart_file.wav` was long (15 seconds), but some noise samples were short.
*   **Action:** The code cut the audio into 15-second chunks. If one was shorter, it would have looped it (though in this final run, we simply truncated to the shortest length).

---

## 5. The "Brain" (The AI Model Logic)

This is what happens inside the `M5_Network.py` and `GFANC_components.py` scripts.

### The Co-Processor (The Decision Maker)
*   **Input:** It takes a buffer of **2048 samples** (128 milliseconds of sound).
*   **Processing:** It passes this sound through 3 Convolutional Layers. These layers look for patterns (shapes) in the sound wave.
*   **Output:** It spits out **15 Numbers** (between 0 and 1). These are the "Ingredients" for the filter.

### The Controller (The Filter Builder)
*   **Action:** It takes the 15 numbers from the Co-Processor.
*   **Mixing:** It goes to a "Database" of 15 pre-learned Sub-Filters (in `Pretrained_Sub_Control_filters_RealPath.mat`).
*   **Logic:** If the Co-Processor outputs `[0.1, 0.9, 0.0...]`, the Controller mixes **10% of Filter A** + **90% of Filter B**.
*   **Result:** This creates one final **Control Filter ($W$)**.

---

## 6. The Cooking Process (Traning & Execution)

The script `reproduce_issue.py` ran the simulation in a loop, block-by-block, mimicking real-time hardware.

### The Loop (Repeated 472 times)
1.  **Get Block:** Grab the next **32 milliseconds** (512 samples) of audio.
2.  **Predict:** Feed the history (128ms) to the AI to generate a Filter.
3.  **Apply:** Use the Filter to mathematically subtract noise from the current block.
    *   Math: $Error = Disturbance - (Filter * Reference)$
4.  **Check:** Did the noise get quieter?
    *   **If YES:** Good job. Tweak the AI slightly to be even better next time (Backpropagation).
    *   **If NO (Amplification):** Stop! The model made it louder. Don't update the AI weights this time. Use the original input as the output to be safe.
5.  **Save:** Append the result to the output list.

### Post-Processing
After the loop finished, the code took the entire result list and ran it through the **100-500 Hz Bandpass Filter** one last time to ensure it sounded clean and smooth.

---

## 7. The Final Result

Three files were generated:
1.  **`gfanc_output.wav`:** The final cleaned audio file. You can listen to this.
2.  **`gfanc_plot.png`:** A graph showing:
    *   **Top (Blue):** The loud, messy input noise.
    *   **Bottom (Green):** The flat, quiet output line.
3.  **`reproduce_issue.py`:** The python script that did all the work.

### Summary for Non-Techies
To get this output, we:
1.  Took a recording of a noisy environment.
2.  Filtered it so we only looked at the "Droning" sounds (100-500Hz).
3.  Fed it into an AI Brain that was trained on thousands of similar sounds.
4.  The AI listened to tiny chunks (32ms) at a time and instantly created a mathematical "Anti-Sound".
5.  We subtracted the Anti-Sound from the Noise.
6.  We saved the silence!
