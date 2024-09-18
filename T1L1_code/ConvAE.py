# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 21:30:46 2024

@author: steve
"""

import numpy as np

data_folder='Task_1_Level_1'
# Load the data from the .npy file
input_data = np.load(r'D:\important\Hensinki_Speech_Challenge_2024\my_project\dataset\%s\new_input.npy'%data_folder)

# Load the data from the .npy file
output_data = np.load(r'D:\important\Hensinki_Speech_Challenge_2024\my_project\dataset\%s\new_output.npy'%data_folder)

#%%
output_noise = input_data-output_data

#%%
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, BatchNormalization
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense,Conv1DTranspose,Conv1D,Flatten,Concatenate,Masking, Reshape, Lambda,Layer,MaxPooling1D,Dropout
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
            
def CNN_model(input_shape):
    inputs = Input(shape=(input_shape[1],))
    encoder = Lambda(lambda x: tf.expand_dims(x, -1))(inputs)
    encoder = Conv1D(filters=8, kernel_size=160, padding="same")(encoder)
    encoder = Conv1D(filters=4, kernel_size=80, padding="same")(encoder)
    # encoder = Conv1D(filters=16, kernel_size=8, padding="same", activation='relu')(encoder)
    encoder = Flatten()(encoder)
    encoder = Dense(units=100)(encoder)     
    decoder = Dense(units=233472)(encoder) 


    model = Model(inputs=inputs, outputs=decoder)
    model.summary()
    return model 

def ConvAE_model(input_shape):
    inputs = Input(shape=(input_shape[1],))
    x = Masking(mask_value=0)(inputs)
    x = Lambda(lambda x: tf.expand_dims(x, -1))(x)
    # Encoder
    x = Conv1D(filters=64, kernel_size=8, activation='linear', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv1D(filters=32, kernel_size=8, activation='linear', padding='same')(x)
    x = BatchNormalization()(x)
    # x = Conv1D(filters=16, kernel_size=4, activation='linear', padding='same',kernel_regularizer=l2(0.01))(x)
    # x = BatchNormalization()(x)
    x = Conv1D(filters=16, kernel_size=4, activation='linear', padding='same')(x)
    
    # Decoder
    # x = Conv1DTranspose(filters=16, kernel_size=2, activation='linear', padding='same',kernel_regularizer=l2(0.01))(encoded)
    # x = BatchNormalization()(x)
    x = Conv1DTranspose(filters=32, kernel_size=4, activation='linear', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv1DTranspose(filters=64, kernel_size=8, activation='linear', padding='same')(x)
    x = BatchNormalization()(x)
    decoded = Conv1DTranspose(filters=1, kernel_size=8, activation='linear', padding='same')(x)

    model = Model(inputs=inputs, outputs=decoded)
    model.summary()
    return model 

def DenseAE_model(input_shape):
    # Input shape is (233472, 1)
    inputs = Input(shape=(input_shape[1],))  # Flattened input (batch_size, steps)
    x = Masking(mask_value=0.0)(inputs)
    # Flatten the input
    x = Lambda(lambda x: tf.reshape(x, (-1, input_shape[1])))(x)  # Flatten (batch_size, steps * features)

    # Encoder
    x = Dense(64, activation='linear')(x)
    x = Dense(32, activation='linear')(x)
    encoded = Dense(16, activation='linear')(x)
    
    # Decoder
    x = Dense(32, activation='linear')(encoded)
    x = Dense(64, activation='linear')(x)
    decoded = Dense(input_shape[1], activation='linear')(x)  # Output shape should match input shape

    # Reshape back to the original dimensions
    decoded = Reshape((input_shape[1],))(decoded)

    model = Model(inputs=inputs, outputs=decoded)
    model.summary()
    return model

#%%
K.clear_session()

model=ConvAE_model(input_data.shape)
learning_rate=0.001 #設定學習速率
adam = Adam(lr=learning_rate) 
model.compile(optimizer=adam,loss="mse") 
earlystopper = EarlyStopping(monitor='val_loss', patience=40, verbose=0) 
checkpoint =ModelCheckpoint(r"D:\important\Hensinki_Speech_Challenge_2024\my_project\model\%s\ConvAE-model-new.hdf5"%data_folder,save_best_only=True) 
callback_list=[earlystopper,checkpoint]  
# history=model.fit(input_data, output_data,epochs=100, batch_size=8,validation_split=0.2,callbacks=callback_list,shuffle=True) 
history=model.fit(input_data, output_noise,epochs=100, batch_size=8,validation_split=0.2,callbacks=callback_list,shuffle=True) 


#%%
#model forecasting result
model=load_model(r"D:\important\Hensinki_Speech_Challenge_2024\my_project\model\%s\ConvAE-model-new.hdf5"%data_folder) #把儲存好的最佳模式讀入
batch_size=8
# pred_data=np.squeeze((model.predict(input_data,batch_size=batch_size))) 

#%%
pred_noise=np.squeeze((model.predict(input_data,batch_size=batch_size))) 
pred_data=input_data-pred_noise
pred_data[(pred_data < 100) & (pred_data > -100)] = 0

#%%
np.save('D:\important\Hensinki_Speech_Challenge_2024\my_project\dataset\%s\ConvAE\pred_data-new.npy'%data_folder, pred_data)

#%%
import matplotlib.pyplot as plt

# Extract loss and val_loss from history
loss = history.history['loss']
val_loss = history.history['val_loss']

# Create the first figure for training loss
plt.figure()
plt.plot(loss, label='Training Loss', color='red')
plt.title("Training Loss")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.legend(loc='upper right')

# Save the first figure
plt.savefig(r'D:\important\Hensinki_Speech_Challenge_2024\my_project\figure\%s\ConvAE\training_loss.png'%data_folder)

# Show the first plot (optional)
plt.show()

# Create the second figure for validation loss
plt.figure()
plt.plot(val_loss, label='Validation Loss', color='blue')
plt.title("Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.legend(loc='upper right')

# Save the second figure
plt.savefig(r'D:\important\Hensinki_Speech_Challenge_2024\my_project\figure\%s\ConvAE\validation_loss.png'%data_folder)

# Show the second plot (optional)
plt.show()
