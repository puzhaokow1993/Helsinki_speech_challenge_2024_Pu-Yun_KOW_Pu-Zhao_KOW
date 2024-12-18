# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 21:41:48 2024

@author: steve
"""

import argparse
import librosa
import numpy as np
import gc
import os
import scipy.io.wavfile as wavfile
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import tensorflow as tf
from scipy import stats
import copy

# os.chdir(os.path.dirname(__file__))

#%%
def main(args, phase="input"):
    # def main(input_wav_dir, task_ID, stage, phase="input"):

    data_folders=['Task_1_Level_1','Task_1_Level_2','Task_1_Level_3','Task_1_Level_4',
                  'Task_1_Level_5','Task_1_Level_6','Task_1_Level_7','Task_2_Level_1',
                  'Task_2_Level_2','Task_2_Level_3','Task_3_Level_1','Task_3_Level_2']
    K.clear_session()
    sr=16000; batch_size=8
    
    if args.task_ID =='T1L1': 
        task_level=data_folders[0]
        max_len=233340
        pred_data=ConvAEfft(args.input_wav_dir, args.stage, task_level, args.task_ID, batch_size, sr, max_len)          
        return pred_data
    
    elif args.task_ID =='T1L2': 
        task_level=data_folders[1]
        max_len=240252
        pred_data=ConvAEfft(args.input_wav_dir, args.stage, task_level, args.task_ID, batch_size, sr, max_len)
        return pred_data
    
    elif args.task_ID =='T1L3':  
        task_level=data_folders[2]
        max_len=223740
        pred_data=ConvAEfft(args.input_wav_dir, args.stage, task_level, args.task_ID, batch_size, sr, max_len)
        return pred_data
    
    elif args.task_ID =='T1L4': 
        task_level=data_folders[3]
        max_len=250236
        underestimation_penalty, overestimation_penalty = 2.0, 1.0  
        pred_data=ConvAEfft2(args.input_wav_dir, args.stage, task_level, args.task_ID, batch_size, sr, max_len,underestimation_penalty, overestimation_penalty, compressed=1)
        return pred_data

    elif args.task_ID =='T1L5':     
        task_level=data_folders[4]
        max_len=220668
        underestimation_penalty, overestimation_penalty = 0, 0 
        pred_data=ConvAEfft2(args.input_wav_dir, args.stage, task_level, args.task_ID, batch_size, sr, max_len, underestimation_penalty, overestimation_penalty, compressed=1)
        return pred_data

    elif args.task_ID =='T1L6': 
        task_level=data_folders[5]
        max_len=238716
        underestimation_penalty, overestimation_penalty = 0, 0         
        pred_data=ConvAEfft2(args.input_wav_dir, args.stage, task_level, args.task_ID, batch_size, sr, max_len, underestimation_penalty, overestimation_penalty, compressed=1)       
        return pred_data

    elif args.task_ID =='T1L7': 
        task_level=data_folders[6]
        max_len=246012
        underestimation_penalty, overestimation_penalty = 0, 0    
        pred_data=ConvAEfft2(args.input_wav_dir, args.stage, task_level, args.task_ID, batch_size, sr, max_len, underestimation_penalty, overestimation_penalty, compressed=1)       
        return pred_data

    elif args.task_ID =='T2L1':   
        task_level=data_folders[7]
        max_len=291517
        pred_data=ConvAEfft3(args.input_wav_dir, args.stage, task_level, args.task_ID, batch_size, sr, max_len, compressed=0)
        return pred_data

    elif args.task_ID =='T2L2':     
        task_level=data_folders[8]
        max_len=301117
        pred_data=ConvAEfft3(args.input_wav_dir, args.stage, task_level, args.task_ID, batch_size, sr, max_len, compressed=0)
        return pred_data

    elif args.task_ID =='T2L3':     
        task_level=data_folders[9]
        max_len=296893
        pred_data=ConvAEfft3(args.input_wav_dir, args.stage, task_level, args.task_ID, batch_size, sr, max_len, compressed=0)
        return pred_data

    elif args.task_ID =='T3L1':     
        task_level=data_folders[10]
        max_len=301117
        pred_data=ConvAEfft4(args.input_wav_dir, args.stage, task_level, args.task_ID, batch_size, sr, max_len, compressed=0)
        return pred_data
    
    elif args.task_ID =='T3L2':     
        task_level=data_folders[11]
        max_len=296893
        pred_data=ConvAEfft4(args.input_wav_dir, args.stage, task_level, args.task_ID, batch_size, sr, max_len, compressed=0)
        return pred_data

def ConvAEfft(input_wav_dir, stage, task_level, task_ID, batch_size, sr, max_len):
    gc.collect()
    # Read all wav files from the specified directory
    input_int16_list, audio_filenames = sampling(input_wav_dir)

    if stage == "training":
        # Modify filenames and load training indices
        new_filenames = np.array([filename.replace('recorded_', '') for filename in audio_filenames])
        indices = np.load("dataset/%s/indices.npy"%task_level)

        select_indices = []
        for filename in new_filenames:
            if filename in indices:
                select_indices.append(int(indices[np.where(indices[:, 0] == filename)[0][0], 1]))
        
        select_indices = np.array(select_indices)
        # Load predicted data from pre-saved model outputs
        pred_data = np.load("dataset/%s/ConvAE/pred_data-fft.npy"%task_level)
        pred_data = pred_data[select_indices, :]
    
    else:
        # Data preprocessing
        log_input_fft_magnitude, input_fft_phase, split_count = fft_log_norm(input_int16_list, max_len)

        # Load deep learning model and make predictions
        model_path = r'model/T1L1-L3/%s/ConvAE-fft.hdf5' %task_ID
        model = load_model(model_path)
        
        # Predict and process the FFT difference
        log_predicted_fft_diff = np.squeeze(model.predict(log_input_fft_magnitude, batch_size=batch_size))
        pred_data = fft_log_denorm(log_input_fft_magnitude, log_predicted_fft_diff, input_fft_phase)
        # merge the padded signal
        pred_data= merge_signals(pred_data, split_count)
    
    # Save the denoised audio back as wav files
    for i in range(len(audio_filenames)):
        new_name = copy.deepcopy(audio_filenames[i])
        if 'recorded' in new_name:
            new_name = new_name.replace('recorded', 'denoise')

        output_wav = os.path.join('output_denoise', task_level, new_name)
        print(new_name)
        # Convert predicted data to int16 format and save as wav file
        audio_int16 = pred_data[i].astype(np.int16)
        save_to_wav(audio_int16, sr, output_wav)
    
    return pred_data

def ConvAEfft2(input_wav_dir, stage, task_level, task_ID, batch_size, sr, max_len,underestimation_penalty=0,overestimation_penalty=0, compressed=0):
    gc.collect()
    # Read all wav files from the specified directory
    input_int16_list, audio_filenames = sampling(input_wav_dir)

    if stage == "training":
        # Process filenames and load indices
        new_filenames = np.array([filename.replace('recorded_', '') for filename in audio_filenames])
        indices = np.load("dataset/%s/indices.npy"%task_level)

        select_indices = []
        for filename in new_filenames:
            if filename in indices:         
                select_indices.append(int(indices[np.where(indices[:, 0] == filename)[0][0], 1]))

        select_indices = np.array(select_indices)
        # Load predicted data for training
        pred_data = np.load("dataset/%s/ConvAE/pred_data-magnitude_phase.npy"%task_level)
        pred_data = pred_data[select_indices, :]

    else:

        # Preprocess the data using FFT normalization
        log_input_fft_magnitude, input_fft_phase, split_count = fft_log_norm(input_int16_list, max_len)
        # Load the deep learning model and predict the FFT difference
        model_path = r'model/T1L4-L7/%s/ConvAE-fft.hdf5'%task_ID
        if underestimation_penalty ==0 and overestimation_penalty==0:     
            model = load_model(model_path)        
        else:
            model = load_model(model_path, custom_objects={'custom_loss': custom_loss(underestimation_penalty,overestimation_penalty)})
        
        if compressed==0:
            log_predicted_diff = np.squeeze(model.predict(log_input_fft_magnitude, batch_size=batch_size))
            log_predicted_fft_magnitude = log_input_fft_magnitude+log_predicted_diff
            del log_input_fft_magnitude
            # Transform the magnitude data
            predicted_fft_magnitude = np.power(10, log_predicted_fft_magnitude)        
        else:
            L = log_input_fft_magnitude.shape[1]
            c = -1; power = 2
            transformed_log_input_fft_magnitude = transform_function(log_input_fft_magnitude, L, c, power)
            del log_input_fft_magnitude
            # Compress the transformed magnitude data
            c = 0.85
            log_input_fft_magnitude = compression_function(transformed_log_input_fft_magnitude, c)
            del transformed_log_input_fft_magnitude
            log_predicted_diff = np.squeeze(model.predict(log_input_fft_magnitude, batch_size=batch_size))
            # Decompress and denormalize the predicted magnitude
            predicted_fft_magnitude = fft_compressed_log_denorm(log_input_fft_magnitude, log_predicted_diff)
        del log_predicted_diff

        predicted_output_fft = predicted_fft_magnitude * np.exp(1j * input_fft_phase)
        # Perform inverse FFT to return to the time domain
        pred_data = np.fft.ifft(predicted_output_fft, axis=1).real  # Take the real part
        del predicted_output_fft
        # merge the padded signal
        pred_data= merge_signals(pred_data, split_count)
        
    # Save the denoised audio back as wav files
    for i in range(len(audio_filenames)):
        new_name = copy.deepcopy(audio_filenames[i])
        if 'recorded' in new_name:
            new_name = new_name.replace('recorded', 'denoise')

        output_wav = os.path.join('output_denoise', task_level, new_name)
        print(new_name)
        # Convert predicted data to int16 format and save as wav file
        audio_int16 = pred_data[i].astype(np.int16)
        save_to_wav(audio_int16, sr, output_wav)

    return pred_data


def ConvAEfft3(input_wav_dir, stage, task_level, task_ID, batch_size, sr, max_len, compressed=0):
# def ConvAEfft3(input_wav_dir, stage, phase, task_level, task_ID, batch_size, sr, max_len):
    gc.collect()
    # Read all wav files from the specified directory
    input_int16_list, audio_filenames = sampling(input_wav_dir)

    if stage == "training":
        # Process filenames and load indices
        new_filenames = np.array([filename.replace('recorded_', '') for filename in audio_filenames])
        indices = np.load("dataset/%s/indices.npy"%task_level)

        if len(indices[0,:])<7:
            new_filenames=[filename[-7:] for filename in new_filenames]

        select_indices = []
        for filename in new_filenames:
            if filename in indices:
                select_indices.append(int(indices[np.where(indices[:, 0] == filename)[0][0], 1]))

        select_indices = np.array(select_indices)
        # Load predicted data for training
        pred_data = np.load("dataset/%s/ConvAE/pred_data-magnitude_phase.npy"%task_level)
        pred_data = pred_data[select_indices, :]

    else:

        # Preprocess the data using FFT normalization
        log_input_fft_magnitude, input_fft_phase, split_count = fft_log_norm(input_int16_list, max_len)
        
        # Load the deep learning model and predict the FFT difference
        model_path = r'model/T2L1-L3/%s/ConvAE-fft.hdf5'%task_ID        
        model = load_model(model_path)
        
        if compressed==0:
            log_predicted_diff = np.squeeze(model.predict(log_input_fft_magnitude, batch_size=batch_size))
            log_predicted_fft_magnitude = log_input_fft_magnitude+log_predicted_diff
            del log_input_fft_magnitude
            # Transform the magnitude data
            predicted_fft_magnitude = np.power(10, log_predicted_fft_magnitude)        
        else:
            L = log_input_fft_magnitude.shape[1]
            c = -1; power = 2
            transformed_log_input_fft_magnitude = transform_function(log_input_fft_magnitude, L, c, power)
            del log_input_fft_magnitude
            # Compress the transformed magnitude data
            c = 0.85
            log_input_fft_magnitude = compression_function(transformed_log_input_fft_magnitude, c)
            del transformed_log_input_fft_magnitude
            log_predicted_diff = np.squeeze(model.predict(log_input_fft_magnitude, batch_size=batch_size))
            # Decompress and denormalize the predicted magnitude
            log_predicted_fft_magnitude = fft_compressed_log_denorm(log_input_fft_magnitude, log_predicted_diff)   
            predicted_fft_magnitude = np.power(10, log_predicted_fft_magnitude)
            
        predicted_output_fft = predicted_fft_magnitude * np.exp(1j * input_fft_phase)
            
        # Perform inverse FFT to return to the time domain
        pred_data = np.fft.ifft(predicted_output_fft, axis=1).real  # Take the real part
        del predicted_output_fft
        # merge the padded signal
        pred_data= merge_signals(pred_data, split_count)
        
    # Save the denoised audio back as wav files
    for i in range(len(audio_filenames)):
        new_name = copy.deepcopy(audio_filenames[i])
        if 'recorded' in new_name:
            new_name = new_name.replace('recorded', 'denoise')

        output_wav = os.path.join('output_denoise', task_level, new_name)
        print(new_name)
        # Convert predicted data to int16 format and save as wav file
        audio_int16 = pred_data[i].astype(np.int16)
        save_to_wav(audio_int16, sr, output_wav)

    return pred_data


def ConvAEfft4(input_wav_dir, stage, task_level, task_ID, batch_size, sr, max_len, compressed=0):
# def ConvAEfft3(input_wav_dir, stage, phase, task_level, task_ID, batch_size, sr, max_len):
    gc.collect()
    # Read all wav files from the specified directory
    input_int16_list, audio_filenames = sampling(input_wav_dir)

    if stage == "training":
        # Process filenames and load indices
        new_filenames = np.array([filename.replace('recorded_', '') for filename in audio_filenames])
        indices = np.load("dataset/%s/indices.npy"%task_level)

        if len(indices[0,:])<7:
            new_filenames=[filename[-7:] for filename in new_filenames]

        select_indices = []
        for filename in new_filenames:
            if filename in indices:
                select_indices.append(int(indices[np.where(indices[:, 0] == filename)[0][0], 1]))

        select_indices = np.array(select_indices)
        # Load predicted data for training
        pred_data = np.load("dataset/%s/ConvAE/pred_data-magnitude_phase.npy"%task_level)
        pred_data = pred_data[select_indices, :]

    else:

        # Preprocess the data using FFT normalization
        log_input_fft_magnitude, input_fft_phase, split_count = fft_log_norm(input_int16_list, max_len)
        
        # Load the deep learning model and predict the FFT difference
        model_path = r'model/T3L1-L2/%s/ConvAE-fft.hdf5'%task_ID        
        model = load_model(model_path)
        
        if compressed==0:
            log_predicted_diff = np.squeeze(model.predict(log_input_fft_magnitude, batch_size=batch_size))
            log_predicted_fft_magnitude = log_input_fft_magnitude+log_predicted_diff
            del log_input_fft_magnitude
            # Transform the magnitude data
            predicted_fft_magnitude = np.power(10, log_predicted_fft_magnitude)        
        else:
            L = log_input_fft_magnitude.shape[1]
            c = -1; power = 2
            transformed_log_input_fft_magnitude = transform_function(log_input_fft_magnitude, L, c, power)
            del log_input_fft_magnitude
            # Compress the transformed magnitude data
            c = 0.85
            log_input_fft_magnitude = compression_function(transformed_log_input_fft_magnitude, c)
            del transformed_log_input_fft_magnitude
            log_predicted_diff = np.squeeze(model.predict(log_input_fft_magnitude, batch_size=batch_size))
            # Decompress and denormalize the predicted magnitude
            log_predicted_fft_magnitude = fft_compressed_log_denorm(log_input_fft_magnitude, log_predicted_diff)
            del log_predicted_diff
            predicted_fft_magnitude = np.power(10, log_predicted_fft_magnitude)
            
        predicted_output_fft = predicted_fft_magnitude * np.exp(1j * input_fft_phase)
            
        # Perform inverse FFT to return to the time domain
        pred_data = np.fft.ifft(predicted_output_fft, axis=1).real  # Take the real part
        del predicted_output_fft
        # merge the padded signal
        pred_data= merge_signals(pred_data, split_count)
        
    # Save the denoised audio back as wav files
    for i in range(len(audio_filenames)):
        new_name = copy.deepcopy(audio_filenames[i])
        if 'recorded' in new_name:
            new_name = new_name.replace('recorded', 'denoise')

        output_wav = os.path.join('output_denoise', task_level, new_name)
        print(new_name)
        # Convert predicted data to int16 format and save as wav file
        audio_int16 = pred_data[i].astype(np.int16)
        save_to_wav(audio_int16, sr, output_wav)
        
    return pred_data

def merge_signals(processed_list, split_count_list):
    merged_signals = []
    idx = 0  # To track where to start merging for each signal

    for i, split_count in enumerate(split_count_list):
        if split_count == 1:
            # If the signal wasn't split, just take it directly
            merged_signal = processed_list[idx,:]
            merged_signals.append(merged_signal)
            idx += 1
        else:
            # Merge the parts back together
            merged_signal = np.concatenate([processed_list[idx + j,:] for j in range(split_count)])
            merged_signals.append(merged_signal)
            idx += split_count  # Move the index by the number of splits

    return merged_signals

# Custom loss function that penalizes underestimations more than overestimations
def custom_loss(underestimation_penalty, overestimation_penalty):
    def loss_function(y_true, y_pred):
        # Implement your custom loss logic here
        # For example:
        diff = y_true - y_pred
        loss = tf.where(diff > 0, underestimation_penalty * diff, overestimation_penalty * tf.abs(diff))
        return tf.reduce_mean(loss)
    return loss_function

def predict_phase3(unwrapped_input_fft_phase, model_path):
    # input_fft_phase=wrap_back(unwrapped_input_fft_phase)
    # Load the pre-trained models
    pca_loaded = joblib.load(f'{model_path}//pca2_model.pkl')
    pca2_loaded = joblib.load(f'{model_path}//pca2-2_model.pkl')
    poly_loaded = joblib.load(f'{model_path}//poly2_transformer.pkl')
    model_loaded = joblib.load(f'{model_path}//linear_regression_model.pkl')
    # Apply PCA to new data
    X_reduced = pca_loaded.transform(unwrapped_input_fft_phase)
    X_poly = poly_loaded.transform(X_reduced)
    
    # Make predictions
    pred_data = model_loaded.predict(X_poly)
    pred_phase = pca2_loaded.inverse_transform(pred_data)    
    return pred_phase

def wrap_back(unwrapped_phase):
    wrapped_back_phase = np.arctan2(np.sin(unwrapped_phase), np.cos(unwrapped_phase))

    # wrapped_back_phase = (unwrapped_phase + np.pi) % (2 * np.pi) - np.pi
    return wrapped_back_phase

def transform_function(x, L, c, power):
    new_x=x+2
    new_x[new_x<0]=0
    return new_x*(1 + c * ((np.array([i for i in range(L)]) - L/2) / L)**power)

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

def fft_compressed_log_denorm(compressed_log_input_fft_magnitude, predicted_diff ):
    log_predicted_fft_magnitude = compressed_log_input_fft_magnitude + predicted_diff
    del compressed_log_input_fft_magnitude, predicted_diff
    predicted_fft_magnitude = np.power(10, log_predicted_fft_magnitude)
    del log_predicted_fft_magnitude
    
    return predicted_fft_magnitude 

def fft_log_norm(input_int16_list,input_len,index=0,lag=0):
    if index==0:
        input_int16_array, split_count= fill_zeros(input_int16_list,input_len)
        try:
            input_fft = np.fft.fft(input_int16_array,axis=1)
        except:
            input_fft = np.fft.fft(input_int16_array)
        # Get the magnitude (absolute value) of the FFT
        input_fft_magnitude = np.abs(input_fft)
        log_input_fft_magnitude = np.log10(input_fft_magnitude+1)
        input_fft_phase = np.angle(input_fft)
        return log_input_fft_magnitude,input_fft_phase, split_count
    else:
        input_int16_array, split_count= fill_zeros(input_int16_list,input_len,1,lag)
        try:
            input_fft = np.fft.fft(input_int16_array,axis=1)
        except:
            input_fft = np.fft.fft(input_int16_array)
        # Get the magnitude (absolute value) of the FFT
        input_fft_magnitude = np.abs(input_fft)
        log_input_fft_magnitude = np.log10(input_fft_magnitude+1)
        input_fft_phase = np.angle(input_fft)
        return log_input_fft_magnitude,np.unwrap(input_fft_phase), split_count
        
def fft_log_denorm(log_input_fft_magnitude,log_predicted_fft_diff,input_fft_phase):
    log_predicted_fft_magnitude = log_input_fft_magnitude + log_predicted_fft_diff 
    predicted_fft_magnitude = np.power(10, log_predicted_fft_magnitude )
    predicted_output_fft = predicted_fft_magnitude * np.exp(1j * input_fft_phase)
    # Perform inverse FFT to transform back to the time domain
    pred_data  = np.fft.ifft(predicted_output_fft, axis=1).real  # Take the real part after IFFT
    return pred_data

def process(audio_file):
    # Load and resample the audio file to 16kHz
    audio, sr = librosa.load(audio_file, sr=16000, mono=True)
    # Convert the audio signal from floating point to 16-bit PCM
    audio_int16 = (audio * np.iinfo(np.int16).max).astype(np.int16)
    return audio_int16 

def sampling(audio_dir):
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    
    audio_int=[]
    for audio_file in audio_files:
        full_path = os.path.join(audio_dir, audio_file)
        audio_int.append(process(full_path))
    
    return audio_int, audio_files

def fill_zeros(raw_signal_list, max_len, index=0, lag=0):
    try:
        processed_list = []
        split_count_list = []
        for signal in raw_signal_list:
            if index != 0:
                if lag!=0:
                    signal = adjust_signal(signal, lag)
    
            # Determine how many parts to split the signal into
            num_splits = int(np.ceil(len(signal) / max_len))
    
            if num_splits == 1:
                # If the signal is less than or equal to max_len, pad it with zeros
                padded_signal = np.pad(signal, (0, max_len - len(signal)), 'constant')
                processed_list.append(padded_signal)
                split_count_list.append(1)  # Record that it wasn't split
    
            else:
                # Split the signal into parts and pad each part separately
                for i in range(num_splits):
                    start_idx = i * max_len
                    end_idx = start_idx + max_len
                    part = signal[start_idx:end_idx]
    
                    # Pad the part with zeros to max_len
                    padded_part = np.pad(part, (0, max_len - len(part)), 'constant')
                    processed_list.append(padded_part)
    
                # Record how many times the signal was split
                split_count_list.append(num_splits)
    except:
        processed_list = []
        split_count_list = []
        signal = adjust_signal(raw_signal_list, lag)
        num_splits = int(np.ceil(len(signal) / max_len))
        if num_splits == 1:
            # If the signal is less than or equal to max_len, pad it with zeros
            padded_signal = np.pad(signal, (0, max_len - len(signal)), 'constant')
            processed_list.append(padded_signal)
            split_count_list.append(1)  # Record that it wasn't split

        else:
            # Split the signal into parts and pad each part separately
            for i in range(num_splits):
                start_idx = i * max_len
                end_idx = start_idx + max_len
                part = signal[start_idx:end_idx]

                # Pad the part with zeros to max_len
                padded_part = np.pad(part, (0, max_len - len(part)), 'constant')
                processed_list.append(padded_part)

            # Record how many times the signal was split
            split_count_list.append(num_splits)

    return processed_list, split_count_list
            
def adjust_signal(signal, lag):
    # Shift the second signal based on the lag
    if lag > 0:
        aligned_signal = np.roll(signal, lag)
        aligned_signal[:lag] = 0  # Set shifted portion to 0 to avoid misalignment
    else:
        aligned_signal = np.roll(signal, lag)
        aligned_signal[lag:] = 0  # Set shifted portion to 0 for negative lag
    
    return aligned_signal

def training_fill_zeros(raw_signal_list, lag, max_len):
    processed_list = []
    
    for raw_signal in raw_signal_list:
        raw_signal= adjust_signal(raw_signal, lag, max_len)
        # Fill the sublist with zeros to match max_len
        padded_sublist = np.pad(raw_signal, (0, max_len - len(raw_signal)), 'constant')
        processed_list.append(padded_sublist)
    
    return processed_list

def save_to_wav(audio_int16, sr, output_wav):
    # Save as WAV file
    wavfile.write(output_wav, sr, audio_int16)
        
#%%
if __name__ == '__main__':
    gc.collect()
    
    parser = argparse.ArgumentParser(description="Denoise audio files")
    parser.add_argument('--input_wav_dir', type=str, required=True, help='Directory containing audio files')
    parser.add_argument('--task_ID', type=str, required=True, help='Task ID')
    parser.add_argument('--stage', type=str, default="testing", help='stage')

    args = parser.parse_args()
    main(args)
    
    


