# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 16:19:18 2024

@author: Steve
"""

import numpy as np
import gc

data_folder='Task_1_Level_4'
input_data = np.load(r'D:\important\Hensinki_Speech_Challenge_2024\my_project\dataset\%s\input_fft_phase.npy'%data_folder)
output_data = np.load(r'D:\important\Hensinki_Speech_Challenge_2024\my_project\dataset\%s\output_fft_phase.npy'%data_folder)

#%%
# Perform FFT along the second axis (23000-dimensional)
input_fft = np.fft.fft(input_data, axis=1)
output_fft = np.fft.fft(output_data, axis=1)
del input_data
input_fft_magnitude = np.abs(input_fft)
output_fft_magnitude = np.abs(output_fft)
del output_fft
input_phase = np.angle(input_fft)
del input_fft
log_input_fft_magnitude = np.log10(input_fft_magnitude+1)
log_output_fft_magnitude = np.log10(output_fft_magnitude+1)
diff = log_output_fft_magnitude - log_input_fft_magnitude
del log_output_fft_magnitude

#%%
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, BatchNormalization
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense,Conv1DTranspose,Conv1D,Flatten,Concatenate,Masking, Reshape, Lambda,Layer,MaxPooling1D,UpSampling1D

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import  l2
import warnings
warnings.filterwarnings("ignore")

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

""" Define ConvAE model """

def ConvAE_model(input_shape):
    inputs = Input(shape=(input_shape[1],))
    # x = Masking(mask_value=0)(inputs)
    x = Lambda(lambda x: tf.expand_dims(x, -1))(inputs)
    # Encoder
    x = Conv1D(filters=64, kernel_size=8, activation='linear', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv1D(filters=32, kernel_size=8, activation='linear', padding='same')(x)
    x = BatchNormalization()(x)
    # x = Conv1D(filters=16, kernel_size=4, activation='linear', padding='same')(x)
    # x = BatchNormalization()(x)
    x = Conv1D(filters=16, kernel_size=4, activation='linear', padding='same')(x)
    
    # Decoder
    # x = Conv1DTranspose(filters=16, kernel_size=4, activation='linear', padding='same')(x)
    # x = BatchNormalization()(x)
    x = Conv1DTranspose(filters=32, kernel_size=8, activation='linear', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv1DTranspose(filters=64, kernel_size=8, activation='linear', padding='same')(x)
    x = BatchNormalization()(x)
    decoded = Conv1DTranspose(filters=1, kernel_size=8, activation='linear', padding='same')(x)

    model = Model(inputs=inputs, outputs=decoded)
    model.summary()
    return model 

#%%
gc.collect()
K.clear_session()

model=ConvAE_model(log_input_fft_magnitude.shape)
learning_rate=0.001 #設定學習速率
adam = Adam(lr=learning_rate) 
model.compile(optimizer=adam,loss=tf.keras.losses.Huber()) 
earlystopper = EarlyStopping(monitor='val_loss', patience=20, verbose=0) 
checkpoint =ModelCheckpoint(r"D:\important\Hensinki_Speech_Challenge_2024\my_project\model\%s\ConvAE-phase.hdf5"%data_folder,save_best_only=True) 
callback_list=[earlystopper,checkpoint]  
# history=model.fit(input_fft_magnitude, output_fft_magnitude,epochs=100, batch_size=8,validation_split=0.2,callbacks=callback_list,shuffle=True) 
history=model.fit(log_input_fft_magnitude, diff, epochs=100,  batch_size=8,validation_split=0.2, callbacks=callback_list,shuffle=True)

#%%
K.clear_session()
#model forecasting result
model=load_model(r"D:\important\Hensinki_Speech_Challenge_2024\my_project\model\%s\ConvAE-phase.hdf5"%data_folder) #把儲存好的最佳模式讀入
batch_size=8
gc.collect()
predicted_diff = np.squeeze((model.predict(log_input_fft_magnitude,batch_size=batch_size))) 
log_predicted_fft_magnitude = log_input_fft_magnitude + predicted_diff
del predicted_diff
predicted_fft_magnitude = np.power(10, log_predicted_fft_magnitude)
del log_predicted_fft_magnitude
predicted_output_fft = predicted_fft_magnitude * np.exp(1j * input_phase)
del predicted_fft_magnitude 
# Perform inverse FFT to transform back to the time domain
pred_data  = np.fft.ifft(predicted_output_fft, axis=1).real  # Take the real part after IFFT
del predicted_output_fft

#%%
# def wrap_to_pi(angle):
#     return (angle + np.pi) % (2 * np.pi) - np.pi


# def normalize_to_pi(data):
#     # Find the minimum and maximum values in the data
#     min_value = np.quantile(data,0.1)
#     max_value = np.quantile(data,0.9)
    
#     # Normalize the data to the range [-pi, pi]
#     normalized_data = -np.pi + ((data - min_value) * (2 * np.pi) / (max_value - min_value))
#     normalized_data = wrap_to_pi(normalized_data)
    
#     return normalized_data

# def normalize_to_pi_quantile(data):

#     # Find the 10th and 90th percentiles
#     min_value = np.quantile(data, 0.1)
#     max_value = np.quantile(data, 0.9)
    
#     # Clip the data so that any values below the 10th percentile are set to min_value
#     # and values above the 90th percentile are set to max_value
#     clipped_data = np.clip(data, min_value, max_value)
    
#     # Normalize the clipped data to the range [-pi, pi]
#     normalized_data = -np.pi + ((clipped_data - min_value) * (2 * np.pi) / (max_value - min_value))
        
#     return normalized_data

# pred_data_  = np.array([normalize_to_pi(wrap_to_pi(pred_data[i,:])) for i in range(len(pred_data))])

#%% save result
np.save(r'D:\important\Hensinki_Speech_Challenge_2024\my_project\dataset\%s\pred_phase2.npy'%data_folder, pred_data)

#%% plot compare if needed
# output_data = np.arctan2(np.sin(output_data), np.cos(output_data))

import matplotlib.pyplot as plt

sample = 100

# plt.figure()
# plt.plot(log_input_fft_magnitude[sample,:], label='output', color='red')
# plt.plot(log_output_fft_magnitude[sample,:], label='predicted', color='blue')
# plt.show()


plt.figure()
plt.plot(output_data[sample,155000:155500], label='output', color='blue')
plt.plot(pred_data[sample,155000:155500], label='predicted', color='red')
plt.show()

# plt.figure()
# plt.plot(input_data[sample,15000:15500], label='input', color='red')
# plt.plot(output_data[sample,15000:15500], label='output', color='blue')
# plt.show()