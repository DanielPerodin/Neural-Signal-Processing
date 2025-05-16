import neo
import numpy as np
import matplotlib.pyplot as plt
import os

# Path to the Data folder
data_folder = 'C:/Users/perodindaniel/Documents/Neural Signal Processing/Data/'

# List all .ncs files in the Data folder
file_paths = [os.path.join(data_folder, file) for file in os.listdir(data_folder) if file.endswith('.ncs')]

# Function to load and process each .ncs file
def process_all_files(file_paths):
    for file_path in file_paths:
        # Load data using Neo
        directory_path = os.path.dirname(file_path)  # Get the directory
        reader = neo.io.NeuralynxIO(dirname=directory_path)
        block = reader.read()

        # Check if block is a list and use the first element
        if isinstance(block, list):
            block = block[0]  # Use the first block in the list
        
        # Check if the block has the expected structure
        if hasattr(block, 'segments') and len(block.segments) > 0:
            signal = block.segments[0].analogsignals[0]
            analyze_channels(signal)
        else:
            print(f"Error: No segments found in block for file {file_path}")
            continue

# Define the functions for spike detection, SNR calculation, and peak-to-trough
def detect_spikes(signal, threshold_factor=5):
    threshold = np.mean(signal) + threshold_factor * np.std(signal) #this is good, and similar to Keaton's, 
    #double check if this is industry standard to add the mean signal and thresholf/NF*std, because old code may have just done NF*std

    #spike detection algo:
    #old code is "spikes = signal > threshold" #defining a spike as anything above the threshold is very sloppy

    #Requirements: 1) signal must be above/below (when negative) the throshold/Noise Floor 2) the spike must be between certain ms (x% tolerance of the standard ms width) wide (that's time from the resting potential to the spike then back to resting potential,
    # we can say that the second resting potential has to be within a 20% range/tolerance of the intial resting potential) - must also go below resting potential if intracellular, above if extracellular.



    spike_times = np.where(spikes)[0]
    #make an image of the spike in matplotlib - once the number of spikes is way more manageable
    return spike_times

def calculate_snr(signal, spike_times, window_size=50):
    snr_values = []
    for spike_time in spike_times:
        # Ensure the window is within bounds
        start = max(0, spike_time - window_size)
        end = min(len(signal), spike_time + window_size)
        
        # Extract the noise segment (around the spike) for calculating noise
        noise_segment = signal[start:end]
        noise_std = np.std(noise_segment)
        
        # Calculate spike amplitude (max - min) around the spike time
        spike_amplitude = np.max(signal[spike_time - window_size: spike_time + window_size]) - np.min(signal[spike_time - window_size: spike_time + window_size])
        
        # Calculate SNR
        if noise_std != 0:
            snr = spike_amplitude / noise_std
        else:
            snr = 0  # If noise is zero, SNR will be zero (can also handle differently if needed)
        
        snr_values.append(snr)
    
    # Return the average SNR for the channel
    return np.mean(snr_values)

def peak_to_trough(signal, spike_times, window_size=50):
    peak_trough_data = []
    for spike_time in spike_times:
        start = max(0, spike_time - window_size)
        end = min(len(signal), spike_time + window_size)
        spike_segment = signal[start:end]
        peak = np.max(spike_segment)
        trough = np.min(spike_segment)
        peak_trough_data.append(peak - trough)
    
    return peak_trough_data

def analyze_channels(signal):
    num_channels = signal.shape[1]
    for channel in range(num_channels):
        channel_signal = signal[:, channel]
        
        spikes = detect_spikes(channel_signal)
        
        # Calculate SNR for the current channel
        snr = calculate_snr(channel_signal, spikes)
        
        # Calculate Peak-to-Trough for the current channel
        peak_trough = peak_to_trough(channel_signal, spikes)
        
        print(f"Channel {channel+1}:")
        print(f"Number of spikes: {len(spikes)}")
        print(f"SNR: {snr}")
        print(f"Peak-to-trough: {np.mean(peak_trough)}")
        print()

def plot_spikes(signal, spikes):
    plt.plot(signal)
    plt.scatter(spikes, signal[spikes], color='r', label="Spikes")
    plt.xlabel('Time (samples)')
    plt.ylabel('Signal Amplitude')
    plt.legend()
    plt.show()

# Process all files in the Data folder
process_all_files(file_paths)
