# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 21:30:46 2024

@author: steve
"""

import numpy as np
import os
path=r"D:\important\Hensinki_Speech_Challenge_2024\my_project"
os.chdir(path)

data_folder='Task_1_Level_5'
# Load the data from the .npy file
input_data = np.load(r'dataset\%s\new_input.npy'%data_folder)

# Load the data from the .npy file
output_data = np.load(r'dataset\%s\new_output.npy'%data_folder)
zero_indices = [np.where(np.abs(input_data[i, :]) < 150)[0] for i in range(input_data.shape[0])]

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

# del input_data, output_data, input_fft, output_fft, input_fft_magnitude

#%%
import gc
gc.collect()

def transform_function(x, L, c, power):
    new_x=x+2
    new_x[new_x<0]=0
    return new_x*(1 + c * ((np.array([i for i in range(L)]) - L/2) / L)**power)

# 假設 L 為 log_input_fft_magnitude 的長度，c 為常數
L = log_input_fft_magnitude.shape[1]
c = -2 # 根據具體情況設定

power = 2
# 應用變換函數

transformed_log_input_fft_magnitude = transform_function(log_input_fft_magnitude, L, c, power)

# del log_input_fft_magnitude

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

#%%
diff = log_output_fft_magnitude - log_input_fft_magnitude
# del transformed_log_input_fft_magnitude

#%%
sample = 10

# Create the first figure for training loss
plt.figure()
plt.semilogy(output_fft_magnitude[sample,:], label='output_fft', color='blue')
plt.semilogy(input_fft_magnitude[sample,:], label='input_fft', color='red')
plt.show()

plt.figure()
plt.semilogy(log_output_fft_magnitude[sample,:], label='output_fft', color='blue')
plt.semilogy(log_input_fft_magnitude[sample,:], label='input_fft', color='red')
plt.show()

plt.figure()
plt.semilogy(log_output_fft_magnitude[sample,:], label='output_fft', color='blue')
plt.semilogy(transformed_log_input_fft_magnitude[sample,:], label='input_fft', color='red')
plt.show()

plt.figure()
plt.semilogy(log_output_fft_magnitude[sample,:], label='output_fft', color='blue')
plt.semilogy(compressed_log_input_fft_magnitude[sample,:], label='input_fft', color='red')
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
    x = Conv1D(filters=16, kernel_size=4, activation='linear', padding='same')(x)
    
    # Decoder
    x = Conv1DTranspose(filters=32, kernel_size=8, activation='linear', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv1DTranspose(filters=64, kernel_size=8, activation='linear', padding='same')(x)
    x = BatchNormalization()(x)
    decoded = Conv1DTranspose(filters=1, kernel_size=8, activation='linear', padding='same')(x)

    model = Model(inputs=inputs, outputs=decoded)
    model.summary()
    return model 

# Custom loss function that penalizes underestimations more than overestimations
def custom_loss(y_true, y_pred):
    # Calculate the difference between true and predicted values
    difference = K.abs(y_true) - K.abs(y_pred)

    # Set different penalties for underestimations and overestimations
    # Underestimation occurs when (y_pred < y_true), i.e., difference > 0
    underestimation_penalty = 0.2  # Larger penalty for underestimation
    overestimation_penalty = 2   # Lower penalty for overestimation

    # Apply penalty based on whether the model is underestimating or overestimating
    loss = tf.where(difference > 0, underestimation_penalty * tf.square(difference), overestimation_penalty * tf.square(difference))

    # Take the mean loss over the batch
    return tf.reduce_mean(loss)

#%%
gc.collect()
K.clear_session()

model=ConvAE_model(log_input_fft_magnitude.shape)
learning_rate=0.01 #設定學習速率
adam = Adam(lr=learning_rate) 
model.compile(optimizer=adam,loss=custom_loss) 
earlystopper = EarlyStopping(monitor='val_loss', patience=20, verbose=0) 
checkpoint =ModelCheckpoint(r"model\%s\ConvAE-model-fft(new2).hdf5"%data_folder,save_best_only=True) 
callback_list=[earlystopper,checkpoint]  
# history=model.fit(input_fft_magnitude, output_fft_magnitude,epochs=100, batch_size=8,validation_split=0.2,callbacks=callback_list,shuffle=True) 
history=model.fit(log_input_fft_magnitude, diff, epochs=100,  batch_size=8,validation_split=0.2, callbacks=callback_list,shuffle=True)

#%%

K.clear_session()
#model forecasting result
model=load_model(r"model\%s\ConvAE-model-fft(new2).hdf5"%data_folder, custom_objects={'custom_loss': custom_loss}) #把儲存好的最佳模式讀入
batch_size=8
gc.collect()
predicted_diff = np.squeeze((model.predict(log_input_fft_magnitude,batch_size=batch_size))) 
log_predicted_fft_magnitude = log_input_fft_magnitude + predicted_diff
del log_input_fft_magnitude, predicted_diff
predicted_fft_magnitude = np.power(10, log_predicted_fft_magnitude )
del log_predicted_fft_magnitude
predicted_output_fft = predicted_fft_magnitude * np.exp(1j * input_fft_phase)
del predicted_fft_magnitude 
# Perform inverse FFT to transform back to the time domain
pred_data  = np.fft.ifft(predicted_output_fft, axis=1).real  # Take the real part after IFFT
del predicted_output_fft

#%% plot figure

sample=500
plt.figure()
plt.plot(input_data[sample,:], label='input', color='red')
plt.plot(output_data[sample,:], label='output', color='blue')
plt.show()

plt.figure()
plt.plot(output_data[sample,:], label='output_data', color='blue')

plt.plot(pred_data[sample,:], label='input_data', color='red')
plt.show()

# plt.figure()
# plt.plot(output_data[sample,:], label='output_data', color='blue')
# plt.plot(new_pred_data[sample,:], label='input_data', color='red')
# plt.show()


#%%
np.save('dataset\%s\ConvAE\pred_data-new2.npy'%data_folder, pred_data)

