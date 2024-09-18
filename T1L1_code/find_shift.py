# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 18:58:42 2024

@author: steve
"""

import numpy as np
# Load the data from the .npy file
input_data = np.load('D:\important\Hensinki_Speech_Challenge_2024\my_project\dataset\Task1_full_input.npy')

# Load the data from the .npy file
output_data = np.load('D:\important\Hensinki_Speech_Challenge_2024\my_project\dataset\Task1_full_output.npy')

#%% Plot both signal data
import matplotlib.pyplot as plt

def plot_signal(signal_data_1,signal_data_2):
    plt.figure(figsize=(10, 4))
    plt.plot(signal_data_2, label='Signal Data 2', color='red')

    plt.plot(signal_data_1, label='Signal Data 1', color='blue')
    plt.title('Comparison of Two Signal Data Sets')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()
    
i=0
plot_signal(input_data[i,:],output_data[i,:])

#%%
i=256
signal_1=output_data[i,:]; signal_2=input_data[i,:]

def adjust_signal(signal_2, signal_1):
    # Compute the cross-correlation between the two signals
    correlation = np.correlate(signal_1 - np.mean(signal_1), signal_2 - np.mean(signal_2), mode='full')
    
    # Find the lag (shift) corresponding to the maximum cross-correlation
    lag = np.argmax(correlation) - (len(signal_1) - 1)
    
    # Shift the second signal based on the lag
    if lag > 0:
        aligned_signal_2 = np.roll(signal_2, lag)
        aligned_signal_2[:lag] = 0  # Set shifted portion to 0 to avoid misalignment
        aligned_signal_1 = signal_1
    else:
        aligned_signal_2 = np.roll(signal_2, lag)
        aligned_signal_2[lag:] = 0  # Set shifted portion to 0 for negative lag
        aligned_signal_1 = signal_1
    
    # Trim both signals to the same length after shifting
    min_length = min(len(aligned_signal_2), len(aligned_signal_1))
    aligned_signal_1 = aligned_signal_1[:min_length]
    aligned_signal_2 = aligned_signal_2[:min_length]
    
    return aligned_signal_2, aligned_signal_1

aligned_signal_2, aligned_signal_1=adjust_signal(signal_2, signal_1)
c=aligned_signal_1 - aligned_signal_2

plot_signal(aligned_signal_2,aligned_signal_1)
