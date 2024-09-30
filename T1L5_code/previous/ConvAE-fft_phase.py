# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 21:30:46 2024

@author: steve
"""

import numpy as np

data_folder='Task_1_Level_5'

input_data = np.load(r'D:\important\Hensinki_Speech_Challenge_2024\my_project\dataset\%s\new_input.npy'%data_folder)
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

del input_fft_magnitude, output_fft_magnitude

#%%
import gc
gc.collect()

def transform_function(x, L, c, power):
    new_x=x+2
    new_x[new_x<0]=0
    return new_x*(1 + c * ((np.array([i for i in range(L)]) - L/2) / L)**power)

# 假設 L 為 log_input_fft_magnitude 的長度，c 為常數
L = log_input_fft_magnitude.shape[1]
c = -1 # 根據具體情況設定
power = 2
# 應用變換函數

transformed_log_input_fft_magnitude = transform_function(log_input_fft_magnitude, L, c, power)
del log_input_fft_magnitude

#%%
gc.collect()
def compression_function(x, c):
    # Step 1: Compute max(x), mean(x), and the first term
    max_x = np.max(x)
    mean_x = np.mean(x)
    term_x = c * (x - mean_x)
    
    # Step 2: Compute max(y), mean(y), and the second term
    max_y = np.max(term_x)

    term_y = max_x - max_y
    
    # Step 3: Calculate z as the sum of the two terms
    z = term_x + term_y
    
    return z

c=0.85
compressed_log_input_fft_magnitude = compression_function(transformed_log_input_fft_magnitude, c)
del transformed_log_input_fft_magnitude

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
gc.collect()
K.clear_session()

#model forecasting result
model=load_model(r"D:\important\Hensinki_Speech_Challenge_2024\my_project\model\%s\ConvAE-model-fft4.hdf5"%data_folder) #fft model
batch_size=8
gc.collect()
# predict magnitude
predicted_diff = np.squeeze((model.predict(compressed_log_input_fft_magnitude,batch_size=batch_size))) 
log_predicted_fft_magnitude = compressed_log_input_fft_magnitude + predicted_diff
del compressed_log_input_fft_magnitude, predicted_diff
predicted_fft_magnitude = np.power(10, log_predicted_fft_magnitude)
del log_predicted_fft_magnitude

# predict phase
predicted_phase = np.load(r'D:\important\Hensinki_Speech_Challenge_2024\my_project\dataset\%s\pred_phase2.npy'%data_folder)

# multiply magtinude with the phase (real and imagenary parts)
predicted_output_fft = predicted_fft_magnitude * np.exp(1j * predicted_phase)
# del predicted_fft_magnitude, predicted_real, predicted_imag  

# Perform inverse FFT to transform back to the time domain
pred_data  = np.fft.ifft(predicted_output_fft, axis=1).real  # Take the real part after IFFT
del predicted_output_fft

#%%
np.save('D:\important\Hensinki_Speech_Challenge_2024\my_project\dataset\%s\ConvAE\pred_data-magnitude_phase.npy'%data_folder, pred_data)

#%%
sample = 1


array1=output_phase[sample,:]
array2=predicted_phase[sample,:]

# Calculate the correlation coefficient
correlation_matrix = np.corrcoef(array1, array2)

# The correlation coefficient is the off-diagonal element
correlation_coefficient = correlation_matrix[0, 1]