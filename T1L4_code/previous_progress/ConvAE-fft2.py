# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 00:03:54 2024

@author: steve
"""

import gc
import numpy as np

data_folder='Task_1_Level_4'
# Load the data from the .npy file
input_data = np.load(r'D:\important\Hensinki_Speech_Challenge_2024\my_project\dataset\%s\ConvAE\pred_data-fft4.npy'%data_folder)
# Load the data from the .npy file
output_data = np.load(r'D:\important\Hensinki_Speech_Challenge_2024\my_project\dataset\%s\new_output.npy'%data_folder)

#%%
import matplotlib.pyplot as plt

# Perform FFT along the second axis (23000-dimensional)
input_fft = np.fft.fft(input_data, axis=1)
output_fft = np.fft.fft(output_data, axis=1)

# Get the magnitude (absolute value) of the FFT
input_fft_magnitude = np.abs(input_fft)
output_fft_magnitude = np.abs(output_fft)
log_input_fft_magnitude = np.log10(input_fft_magnitude+1)
log_output_fft_magnitude = np.log10(output_fft_magnitude+1)

# log_fft_magnitude_diff = log_output_fft_magnitude - log_input_fft_magnitude
input_fft_phase = np.angle(input_fft)
diff = log_output_fft_magnitude -log_input_fft_magnitude 

#%%
import matplotlib.pyplot as plt

sample = 0

# Create the first figure for training loss
plt.figure()
plt.semilogy(input_fft_magnitude[sample,:], label='input', color='red')
plt.semilogy(output_fft_magnitude[sample,:], label='output', color='blue')
plt.show()

# Create the first figure for training loss
plt.figure()
plt.semilogy(log_input_fft_magnitude[sample,:], label='input', color='red')
plt.semilogy(log_output_fft_magnitude[sample,:], label='output', color='blue')
plt.show()

#%%
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, BatchNormalization
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense,Conv1DTranspose,Conv1D,Flatten,Concatenate,Masking, Reshape, Lambda,Layer,MaxPooling1D,UpSampling1D

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import  l2

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

""" Define DNN model """
            

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
model.compile(optimizer=adam,loss="mse") 
earlystopper = EarlyStopping(monitor='val_loss', patience=20, verbose=0) 
checkpoint =ModelCheckpoint(r"D:\important\Hensinki_Speech_Challenge_2024\my_project\model\%s\ConvAE-model-fft2.hdf5"%data_folder,save_best_only=True) 
callback_list=[earlystopper,checkpoint]  
# history=model.fit(input_fft_magnitude, output_fft_magnitude,epochs=100, batch_size=8,validation_split=0.2,callbacks=callback_list,shuffle=True) 
history=model.fit(log_input_fft_magnitude, diff, epochs=100,  batch_size=8,validation_split=0.2, callbacks=callback_list,shuffle=True)

#%%

K.clear_session()
#model forecasting result
model=load_model(r"D:\important\Hensinki_Speech_Challenge_2024\my_project\model\%s\ConvAE-model-fft2.hdf5"%data_folder) #把儲存好的最佳模式讀入
batch_size=8
gc.collect()
predicted_diff = np.squeeze((model.predict(log_input_fft_magnitude,batch_size=batch_size))) 
log_predicted_fft_magnitude = log_input_fft_magnitude + predicted_diff
del log_input_fft_magnitude, predicted_diff
predicted_fft_magnitude = np.power(10, log_predicted_fft_magnitude)
del log_predicted_fft_magnitude
predicted_output_fft = predicted_fft_magnitude * np.exp(1j * input_fft_phase)
del predicted_fft_magnitude 
# Perform inverse FFT to transform back to the time domain
pred_data  = np.fft.ifft(predicted_output_fft, axis=1).real  # Take the real part after IFFT
del predicted_output_fft

#%% data postprocessing
raw_input_data = np.load(r'D:\important\Hensinki_Speech_Challenge_2024\my_project\dataset\%s\new_input.npy'%data_folder)
zero_indices = [np.where(np.abs(raw_input_data[i, :]) < 150)[0] for i in range(raw_input_data.shape[0])]
new_pred_data = np.copy(pred_data)
for i in range(len(zero_indices)):
    new_pred_data[i, zero_indices[i]] = 0

#%%
sample = 1
# Create the first figure for training loss
plt.figure()
plt.plot(output_data[sample,:], label='output_data', color='blue')
plt.plot(pred_data[sample,:], label='pred_data', color='red')
plt.show()

plt.figure()
plt.plot(output_data[sample,:], label='output_data', color='blue')
plt.plot(new_pred_data[sample,:], label='pred_data', color='red')
plt.show()

plt.figure()
plt.plot(output_data[sample,:], label='output_data', color='blue')
plt.plot(input_data[sample,:], label='input_data', color='red')
plt.show()

#%%
np.save('D:\important\Hensinki_Speech_Challenge_2024\my_project\dataset\%s\ConvAE\pred_data-fft2.npy'%data_folder, pred_data)

