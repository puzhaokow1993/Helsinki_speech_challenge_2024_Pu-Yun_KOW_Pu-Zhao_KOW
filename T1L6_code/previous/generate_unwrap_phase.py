# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 19:57:36 2024

@author: steve
"""

import numpy as np

data_folder='Task_1_Level_6'
# Load the data from the .npy file
input_data = np.load(r'D:\important\Hensinki_Speech_Challenge_2024\my_project\dataset\%s\new_input.npy'%data_folder)

# Load the data from the .npy file
output_data = np.load(r'D:\important\Hensinki_Speech_Challenge_2024\my_project\dataset\%s\new_output.npy'%data_folder)

#%%
import matplotlib.pyplot as plt

# Perform FFT along the second axis (23000-dimensional)
input_fft = np.fft.fft(input_data, axis=1)
output_fft = np.fft.fft(output_data, axis=1)
output_phase=np.angle(output_fft)

del input_data, output_data

# Get the magnitude (absolute value) of the FFT
input_fft_magnitude = np.abs(input_fft)
output_fft_magnitude = np.abs(output_fft)
log_input_fft_magnitude = np.log10(input_fft_magnitude+1)
log_output_fft_magnitude = np.log10(output_fft_magnitude+1)

unwrapped_input_fft_phase = np.unwrap(np.angle(input_fft))
unwrapped_output_fft_phase = np.unwrap(np.angle(output_fft))

del input_fft, output_fft
unwrapped_phase_difference = unwrapped_output_fft_phase - unwrapped_input_fft_phase

#%%
np.save(r'D:\important\Hensinki_Speech_Challenge_2024\my_project\dataset\%s\unwrapped_input_fft_phase.npy'%data_folder, unwrapped_input_fft_phase)
np.save(r'D:\important\Hensinki_Speech_Challenge_2024\my_project\dataset\%s\unwrapped_output_fft_phase.npy'%data_folder, unwrapped_output_fft_phase)
