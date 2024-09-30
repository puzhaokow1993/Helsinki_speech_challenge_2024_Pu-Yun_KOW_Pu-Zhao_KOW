# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 21:30:46 2024

@author: steve
"""

import numpy as np

data_folder='Task_3_Level_1'

input_data = np.load(r'D:\important\Hensinki_Speech_Challenge_2024\my_project\dataset\%s\new_input.npy'%data_folder)
output_data = np.load(r'D:\important\Hensinki_Speech_Challenge_2024\my_project\dataset\%s\new_output.npy'%data_folder)

#%%

# Perform FFT along the second axis (23000-dimensional)
input_fft = np.fft.fft(input_data, axis=1)
output_fft = np.fft.fft(output_data, axis=1)

del input_data, output_data

# Get the magnitude (absolute value) of the FFT
input_fft_magnitude = np.abs(input_fft)
output_fft_magnitude = np.abs(output_fft)
log_input_fft_magnitude = np.log10(input_fft_magnitude+1)
log_output_fft_magnitude = np.log10(output_fft_magnitude+1)

del input_fft_magnitude, output_fft_magnitude

#%%
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K


physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Available GPUs:", physical_devices)

# Check if TensorFlow is using GPU
if len(physical_devices) > 0:
    print("TensorFlow is using GPU.")
else:
    print("TensorFlow is using CPU.")
    
# Get TensorFlow build information
build_info = tf.sysconfig.get_build_info()

# Print CUDA and cuDNN versions
print(f"CUDA version: {build_info.get('cuda_version', 'Not found')}")
print(f"cuDNN version: {build_info.get('cudnn_version', 'Not found')}")

#%%
import gc
gc.collect()
K.clear_session()

#model forecasting result
model=load_model(r"D:\important\Hensinki_Speech_Challenge_2024\my_project\model\%s\ConvAE-fft.hdf5"%data_folder) #fft model
batch_size=8
log_predicted_fft_diff = np.squeeze((model.predict(log_input_fft_magnitude,batch_size=batch_size))) 
# predicted_fft_diff = np.exp(log_predicted_fft_diff)
log_predicted_fft_magnitude = log_input_fft_magnitude + log_predicted_fft_diff 
predicted_fft_magnitude = np.power(10, log_predicted_fft_magnitude )

# predict phase
predicted_phase = np.load(r'D:\important\Hensinki_Speech_Challenge_2024\my_project\dataset\%s\pred_phase.npy'%data_folder)

# multiply magtinude with the phase (real and imagenary parts)
predicted_output_fft = predicted_fft_magnitude * np.exp(1j * predicted_phase)
del predicted_fft_magnitude

# Perform inverse FFT to transform back to the time domain
pred_data  = np.fft.ifft(predicted_output_fft, axis=1).real  # Take the real part after IFFT
del predicted_output_fft

#%%
np.save('D:\important\Hensinki_Speech_Challenge_2024\my_project\dataset\%s\ConvAE\pred_data-magnitude_phase.npy'%data_folder, pred_data)

#%%
# sample = 1

# output_phase=np.angle(output_fft)

# array1=output_phase[sample,:]
# array2=predicted_phase[sample,:]

# # Calculate the correlation coefficient
# correlation_matrix = np.corrcoef(array1, array2)

# # The correlation coefficient is the off-diagonal element
# correlation_coefficient = correlation_matrix[0, 1]