# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 21:30:46 2024

@author: steve
"""

import numpy as np

data_folder='Task_1_Level_4'

input_data = np.load(r'D:\important\Hensinki_Speech_Challenge_2024\my_project\dataset\%s\new_input.npy'%data_folder)
output_data = np.load(r'D:\important\Hensinki_Speech_Challenge_2024\my_project\dataset\%s\new_output.npy'%data_folder)
input_fft_magnitude = np.load(r'D:\important\Hensinki_Speech_Challenge_2024\my_project\dataset\%s\input_fft_magnitude.npy'%data_folder)
# output_fft_magnitude = np.load(r'D:\important\Hensinki_Speech_Challenge_2024\my_project\dataset\%s\output_fft_magnitude.npy'%data_folder)
input_phase = np.load(r'D:\important\Hensinki_Speech_Challenge_2024\my_project\dataset\%s\input_fft_phase.npy'%data_folder)
# output_phase = np.load(r'D:\important\Hensinki_Speech_Challenge_2024\my_project\dataset\%s\output_fft_phase.npy'%data_folder)

indices = [input_fft_magnitude[sample,:]>np.quantile(input_fft_magnitude[sample,:],0.99) for sample in range(len(input_fft_magnitude))]
new_magnitude = np.array([input_fft_magnitude[sample,:]*indices[sample] for sample in range(len(input_fft_magnitude))])
new_input_fft = new_magnitude * np.exp(1j * input_phase)
# Perform inverse FFT to transform back to the time domain
new_input = np.fft.ifft(new_input_fft, axis=1).real

#%%
import matplotlib.pyplot as plt

sample = 500
# plt.figure()
# plt.plot(input_fft_magnitude[sample,:], label='predicted', color='red')
# plt.plot(output_fft_magnitude[sample,:], label='output', color='blue')
# plt.show()

plt.figure()
plt.plot(new_input[sample,:], label='predicted', color='red')
plt.plot(output_data[sample,:], label='output', color='blue')
plt.show()

plt.figure()
plt.plot(input_data[sample,:], label='predicted', color='red')
plt.plot(output_data[sample,:], label='output', color='blue')
plt.show()


#%% save output
np.save('D:\important\Hensinki_Speech_Challenge_2024\my_project\dataset\%s\ConvAE\pred_data-filter.npy'%data_folder, new_input)

