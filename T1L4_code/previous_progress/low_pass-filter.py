# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 16:06:57 2024

@author: steve
"""


import numpy as np
from scipy.fft import fft, ifft, fftfreq

# Load the input data (assuming it's a 1D signal or a batch of 1D signals)
data_folder = 'Task_1_Level_4'
input_data = np.load(r'D:\important\Hensinki_Speech_Challenge_2024\my_project\dataset\%s\new_input.npy' % data_folder)
output_data = np.load(r'D:\important\Hensinki_Speech_Challenge_2024\my_project\dataset\%s\new_output.npy' % data_folder)
input_phase = np.load(r'D:\important\Hensinki_Speech_Challenge_2024\my_project\dataset\%s\input_fft_phase.npy'%data_folder)
output_phase = np.load(r'D:\important\Hensinki_Speech_Challenge_2024\my_project\dataset\%s\output_fft_phase.npy'%data_folder)

#%%
# Let's assume `input_data` is a 2D array where each row is a separate signal, for example (n_samples, signal_length)

# Define a low-pass filter (cutoff frequency)
def low_pass_filter(data, cutoff_freq, sampling_rate):
    n_samples, signal_length = data.shape
    filtered_data = np.zeros_like(data)
    
    for i in range(n_samples):
        # Perform Fourier Transform on each signal (row)
        signal = data[i]
        yf = fft(signal)
        xf = fftfreq(signal_length, 1 / sampling_rate)[:signal_length // 2]
        
        # Extract magnitude and phase
        magnitude = np.abs(yf)
        phase = np.angle(yf)
        
        # Apply low-pass filtering to the magnitude (zero out high frequencies)
        cutoff_index = np.where(xf > cutoff_freq)[0][0]
        filtered_magnitude = np.copy(magnitude)
        filtered_magnitude[cutoff_index:] = 0
        filtered_magnitude[-cutoff_index:] = 0
        
        # Combine filtered magnitude with the original phase
        filtered_yf = filtered_magnitude * np.exp(1j * phase)
        
        # Perform inverse Fourier Transform to reconstruct the signal
        filtered_signal = ifft(filtered_yf)
        
        # Store the real part of the reconstructed signal (the imaginary part should be close to zero)
        filtered_data[i] = np.real(filtered_signal)
    
    return filtered_data

# Example usage:
sampling_rate = 16000  # Set this according to your actual data
cutoff_freq = 100  # Define the cutoff frequency for low-pass filtering

input_new_phase=low_pass_filter(input_phase, cutoff_freq, sampling_rate)

#%%
import matplotlib.pyplot as plt

sample = 0

plt.figure()
plt.plot(output_phase[sample,:], label='output', color='blue')
plt.plot(input_new_phase[sample,:], label='predicted', color='red')
plt.show()

plt.figure()
plt.plot(output_phase[sample,:], label='output', color='blue')
plt.plot(input_phase[sample,:], label='predicted', color='red')
plt.show()