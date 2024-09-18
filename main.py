# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 21:41:48 2024

@author: steve
"""


import librosa
import numpy as np
import os
import scipy.io.wavfile as wavfile
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K


os.chdir(os.path.dirname(__file__))

#%%

input_wav_dir = r'D:\important\Hensinki_Speech_Challenge_2024\my_project\raw_dataset\Task_1_Level_1\Recorded'
task_ID ='T1L1'

def main(input_wav_dir, task_ID):
    data_folders=['Task_1_Level_1','Task_1_Level_2','Task_1_Level_3','Task_1_Level_4',
                  'Task_1_Level_5','Task_1_Level_6','Task_1_Level_7','Task_2_Level_1',
                  'Task_2_Level_2','Task_2_Level_3','Task_3_Level_1','Task_3_Level_2']
    K.clear_session()
    sr=16000; batch_size=8

    if task_ID =='T1L1': 
        # read all wav files from specific directory
        input_int16_list, audio_filenames = sampling(input_wav_dir)
        # data preprocessing
        log_input_fft_magnitude,input_fft_phase = fft_log_norm(input_int16_list,233340)
        # read deep learning model and predict
        model_path=r'model\%s\ConvAE-model-fft3.hdf5'%data_folders[1]
        model = load_model(model_path)
        log_predicted_fft_diff = np.squeeze((model.predict(log_input_fft_magnitude,batch_size=batch_size))) 
        pred_data = fft_log_denorm(log_input_fft_magnitude,log_predicted_fft_diff,input_fft_phase)
        pred_data = [pred_data[i,:len(input_int16_list[i])] for i in range(len(pred_data))]   
        # save wav to the folder
        for i in range(len(audio_filenames)):
            old_name = audio_filenames[i]
            new_name = old_name.replace('recorded', 'denoise')
            output_wav = 'output_denoise\%s\%s'%(data_folders[0],new_name)
            audio_int16= pred_data[i].astype(np.int16)
            save_to_wav(audio_int16, sr, output_wav)
        
        return pred_data
    
    elif task_ID =='T1L2': 
        # read all wav files from specific directory
        input_int16_list, audio_filenames = sampling(input_wav_dir)
        # data preprocessing
        log_input_fft_magnitude,input_fft_phase = fft_log_norm(input_int16_list,233340)
        # read deep learning model and predict
        model_path=r'model\%s\ConvAE-model-fft3.hdf5'%data_folders[1]
        model = load_model(model_path)
        log_predicted_fft_diff = np.squeeze((model.predict(log_input_fft_magnitude,batch_size=batch_size))) 
        pred_data = fft_log_denorm(log_input_fft_magnitude,log_predicted_fft_diff,input_fft_phase)
        pred_data = [pred_data[i,:len(input_int16_list[i])] for i in range(len(pred_data))]   
        # save wav to the folder
        for i in range(len(audio_filenames)):
            old_name = audio_filenames[i]
            new_name = old_name.replace('recorded', 'denoise')
            output_wav = 'output_denoise\%s\%s'%(data_folders[0],new_name)
            audio_int16= pred_data[i].astype(np.int16)
            save_to_wav(audio_int16, sr, output_wav)
        
        return pred_data
    
    elif task_ID =='T1L3': 
        # read all wav files from specific directory
        input_int16_list, audio_filenames = sampling(input_wav_dir)
        # data preprocessing
        log_input_fft_magnitude,input_fft_phase = fft_log_norm(input_int16_list,233340)
        # read deep learning model and predict
        model_path=r'model\%s\ConvAE-model-fft3.hdf5'%data_folders[1]
        model = load_model(model_path)
        log_predicted_fft_diff = np.squeeze((model.predict(log_input_fft_magnitude,batch_size=batch_size))) 
        pred_data = fft_log_denorm(log_input_fft_magnitude,log_predicted_fft_diff,input_fft_phase)
        pred_data = [pred_data[i,:len(input_int16_list[i])] for i in range(len(pred_data))]   
        # save wav to the folder
        for i in range(len(audio_filenames)):
            old_name = audio_filenames[i]
            new_name = old_name.replace('recorded', 'denoise')
            output_wav = 'output_denoise\%s\%s'%(data_folders[0],new_name)
            audio_int16= pred_data[i].astype(np.int16)
            save_to_wav(audio_int16, sr, output_wav)
        
        return pred_data
    
    elif task_ID =='T1L4': 
        # read all wav files from specific directory
        input_int16_list, audio_filenames = sampling(input_wav_dir)
        input_int16_array, log_input_fft_magnitude,input_fft_phase = fft_log_norm(input_int16_list,233340,1)
        # data preprocessing
        L = log_input_fft_magnitude.shape[1]; c = -1; power = 2
        transformed_log_input_fft_magnitude = transform_function(log_input_fft_magnitude, L, c, power)
        del log_input_fft_magnitude; c=0.85
        compressed_log_input_fft_magnitude = compression_function(transformed_log_input_fft_magnitude, c)
        del transformed_log_input_fft_magnitude
        # read deep learning model and predict
        model_path=r'model\%s\ConvAE-model-fft4.hdf5'%data_folders[1]
        model = load_model(model_path)
        predicted_diff = np.squeeze((model.predict(compressed_log_input_fft_magnitude,batch_size=batch_size))) 
        predicted_fft_magnitude = fft_compressed_log_denorm(compressed_log_input_fft_magnitude, predicted_diff )
        del predicted_diff
        # pred_phase
        pred_phase = predict_phase(input_int16_array, input_int16_list, model_path='model/Task_1_Level_4/')
        # multiply magtinude with the phase (real and imagenary parts)
        pred_data =  predicted_fft_magnitude * np.exp(1j * pred_phase)
        # Adjust the length of predictions based on input_int16_list
        pred_data = [pred_data[i, :len(input_int16_list[i])] for i in range(len(pred_data))]
        # save wav to the folder
        for i in range(len(audio_filenames)):
            old_name = audio_filenames[i]
            new_name = old_name.replace('recorded', 'denoise')
            output_wav = 'output_denoise\%s\%s'%(data_folders[0],new_name)
            audio_int16= pred_data[i].astype(np.int16)
            save_to_wav(audio_int16, sr, output_wav)
        
        return pred_data

    elif task_ID =='T1L5': 
        # read all wav files from specific directory
        input_int16_list, audio_filenames = sampling(input_wav_dir)
        input_int16_array, log_input_fft_magnitude,input_fft_phase = fft_log_norm(input_int16_list,233340,1)
        # data preprocessing
        L = log_input_fft_magnitude.shape[1]; c = -1; power = 2
        transformed_log_input_fft_magnitude = transform_function(log_input_fft_magnitude, L, c, power)
        del log_input_fft_magnitude; c=0.85
        compressed_log_input_fft_magnitude = compression_function(transformed_log_input_fft_magnitude, c)
        del transformed_log_input_fft_magnitude
        # read deep learning model and predict
        model_path=r'model\%s\ConvAE-model-fft4.hdf5'%data_folders[1]
        model = load_model(model_path)
        predicted_diff = np.squeeze((model.predict(compressed_log_input_fft_magnitude,batch_size=batch_size))) 
        predicted_fft_magnitude = fft_compressed_log_denorm(compressed_log_input_fft_magnitude, predicted_diff )
        del predicted_diff
        # pred_phase
        pred_phase = predict_phase(input_int16_array, input_int16_list, model_path='model/Task_1_Level_5/')
        # multiply magtinude with the phase (real and imagenary parts)
        pred_data =  predicted_fft_magnitude * np.exp(1j * pred_phase)
        # Adjust the length of predictions based on input_int16_list
        pred_data = [pred_data[i, :len(input_int16_list[i])] for i in range(len(pred_data))]
        # save wav to the folder
        for i in range(len(audio_filenames)):
            old_name = audio_filenames[i]
            new_name = old_name.replace('recorded', 'denoise')
            output_wav = 'output_denoise\%s\%s'%(data_folders[0],new_name)
            audio_int16= pred_data[i].astype(np.int16)
            save_to_wav(audio_int16, sr, output_wav)
        
        return pred_data

    elif task_ID =='T1L6': 
        # read all wav files from specific directory
        input_int16_list, audio_filenames = sampling(input_wav_dir)
        input_int16_array, log_input_fft_magnitude,input_fft_phase = fft_log_norm(input_int16_list,233340,1)
        # data preprocessing
        L = log_input_fft_magnitude.shape[1]; c = -1; power = 2
        transformed_log_input_fft_magnitude = transform_function(log_input_fft_magnitude, L, c, power)
        del log_input_fft_magnitude; c=0.85
        compressed_log_input_fft_magnitude = compression_function(transformed_log_input_fft_magnitude, c)
        del transformed_log_input_fft_magnitude
        # read deep learning model and predict
        model_path=r'model\%s\ConvAE-model-fft4.hdf5'%data_folders[1]
        model = load_model(model_path)
        predicted_diff = np.squeeze((model.predict(compressed_log_input_fft_magnitude,batch_size=batch_size))) 
        predicted_fft_magnitude = fft_compressed_log_denorm(compressed_log_input_fft_magnitude, predicted_diff )
        del predicted_diff
        # pred_phase
        pred_phase = predict_phase(input_int16_array, input_int16_list, model_path='model/Task_1_Level_5/')
        # multiply magtinude with the phase (real and imagenary parts)
        pred_data =  predicted_fft_magnitude * np.exp(1j * pred_phase)
        # Adjust the length of predictions based on input_int16_list
        pred_data = [pred_data[i, :len(input_int16_list[i])] for i in range(len(pred_data))]
        # save wav to the folder
        for i in range(len(audio_filenames)):
            old_name = audio_filenames[i]
            new_name = old_name.replace('recorded', 'denoise')
            output_wav = 'output_denoise\%s\%s'%(data_folders[0],new_name)
            audio_int16= pred_data[i].astype(np.int16)
            save_to_wav(audio_int16, sr, output_wav)
        
        return pred_data

    elif task_ID =='T1L7': 
        # read all wav files from specific directory
        input_int16_list, audio_filenames = sampling(input_wav_dir)
        input_int16_array, log_input_fft_magnitude,input_fft_phase = fft_log_norm(input_int16_list,233340,1)
        # data preprocessing
        L = log_input_fft_magnitude.shape[1]; c = -1; power = 2
        transformed_log_input_fft_magnitude = transform_function(log_input_fft_magnitude, L, c, power)
        del log_input_fft_magnitude; c=0.85
        compressed_log_input_fft_magnitude = compression_function(transformed_log_input_fft_magnitude, c)
        del transformed_log_input_fft_magnitude
        # read deep learning model and predict
        model_path=r'model\%s\ConvAE-model-fft4.hdf5'%data_folders[1]
        model = load_model(model_path)
        predicted_diff = np.squeeze((model.predict(compressed_log_input_fft_magnitude,batch_size=batch_size))) 
        predicted_fft_magnitude = fft_compressed_log_denorm(compressed_log_input_fft_magnitude, predicted_diff )
        del predicted_diff
        # pred_phase
        pred_phase = predict_phase(input_int16_array, input_int16_list, model_path='model/Task_1_Level_5/')
        # multiply magtinude with the phase (real and imagenary parts)
        pred_data =  predicted_fft_magnitude * np.exp(1j * pred_phase)
        # Adjust the length of predictions based on input_int16_list
        pred_data = [pred_data[i, :len(input_int16_list[i])] for i in range(len(pred_data))]
        # save wav to the folder
        for i in range(len(audio_filenames)):
            old_name = audio_filenames[i]
            new_name = old_name.replace('recorded', 'denoise')
            output_wav = 'output_denoise\%s\%s'%(data_folders[0],new_name)
            audio_int16= pred_data[i].astype(np.int16)
            save_to_wav(audio_int16, sr, output_wav)
        
        return pred_data

    elif task_ID =='T2L1': 
        # read all wav files from specific directory
        input_int16_list, audio_filenames = sampling(input_wav_dir)
        # data preprocessing
        log_input_fft_magnitude,input_fft_phase = fft_log_norm(input_int16_list,233340)
        # read deep learning model and predict
        model_path=r'model\%s\ConvAE-model-fft3.hdf5'%data_folders[1]
        model = load_model(model_path)
        log_predicted_fft_diff = np.squeeze((model.predict(log_input_fft_magnitude,batch_size=batch_size))) 
        
        # pred_phase
        pred_phase = predict_phase(input_int16_array, input_int16_list, model_path='model/Task_1_Level_5/')
        pred_data = fft_log_denorm(log_input_fft_magnitude,log_predicted_fft_diff,pred_phase)

        # multiply magtinude with the phase (real and imagenary parts)
        pred_data =  predicted_fft_magnitude * np.exp(1j * pred_phase)
        # Adjust the length of predictions based on input_int16_list
        pred_data = [pred_data[i, :len(input_int16_list[i])] for i in range(len(pred_data))]
        # save wav to the folder
        for i in range(len(audio_filenames)):
            old_name = audio_filenames[i]
            new_name = old_name.replace('recorded', 'denoise')
            output_wav = 'output_denoise\%s\%s'%(data_folders[0],new_name)
            audio_int16= pred_data[i].astype(np.int16)
            save_to_wav(audio_int16, sr, output_wav)
        
        return pred_data


def predict_phase(input_int16_array, input_int16_list, model_path):
    # Load the pre-trained models
    pca_loaded = joblib.load(f'{model_path}pca_model.pkl')
    poly_loaded = joblib.load(f'{model_path}poly_transformer.pkl')
    model_loaded = joblib.load(f'{model_path}polynomial_regression_model.pkl')
    # Apply PCA to new data
    X_reduced = pca_loaded.transform(input_int16_array)
    # Apply polynomial transformation
    X_poly = poly_loaded.transform(X_reduced)
    # Make predictions
    pred_difference = model_loaded.predict(X_poly)
    # Add the predicted difference to the original data
    pred_phase = input_int16_array + pred_difference
    return pred_phase

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

def fft_log_norm(input_int16_list,input_len,index=0):
    input_int16_array= np.array(fill_zeros(input_int16_list,input_len))
    input_fft = np.fft.fft(input_int16_array,axis=1)
    # Get the magnitude (absolute value) of the FFT
    input_fft_magnitude = np.abs(input_fft)
    log_input_fft_magnitude = np.log10(input_fft_magnitude+1)
    input_fft_phase = np.angle(input_fft)
    if index==0:
        return log_input_fft_magnitude,input_fft_phase
    else:
        return input_int16_array, log_input_fft_magnitude,input_fft_phase
       
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
    for audio_file in sorted(audio_files):
        full_path = os.path.join(audio_dir, audio_file)
        audio_int.append(process(full_path))
    
    return audio_int, audio_files

def find_zero_segments(arr):
    """Find contiguous zero segments from the beginning and end."""
    # Find zeros from the beginning
    start_zero_count = 0
    for i in range(len(arr)):
        if arr[i] == 0:
            start_zero_count += 1
        else:
            break
    
    # Find zeros from the end
    end_zero_count = 0
    for i in range(len(arr) - 1, -1, -1):
        if arr[i] == 0:
            end_zero_count += 1
        else:
            break
    
    return start_zero_count, end_zero_count

def trim_based_on_start(arr):
    """Trim the largest contiguous zero segment based on the start of the array."""
    start_zero_count, end_zero_count = find_zero_segments(arr)
    
    if start_zero_count > 0 and arr[0] == 0:
        # Remove zeros from the start if the first element is zero
        trimmed_arr = arr[start_zero_count:]
    else:
        # Remove zeros from the end if the first element is not zero
        trimmed_arr = arr[:-end_zero_count] if end_zero_count > 0 else arr
    
    return trimmed_arr

def fill_zeros(list1, max_len):
    processed_list = []
    
    for sublist in list1:
        if len(sublist) > max_len:
            # Trim based on whether the first element is zero or not
            trimmed_sublist = trim_based_on_start(sublist)
            if len(trimmed_sublist) > max_len:
                # If trimmed length still exceeds max_len, truncate
                trimmed_sublist = trimmed_sublist[:max_len]
            # Fill the sublist with zeros to match max_len
            padded_sublist = np.pad(trimmed_sublist, (0, max_len - len(trimmed_sublist)), 'constant')

            processed_list.append(padded_sublist)
        else:
            # Fill the sublist with zeros to match max_len
            padded_sublist = np.pad(sublist, (0, max_len - len(sublist)), 'constant')
            processed_list.append(padded_sublist)
    
    return processed_list


def save_to_wav(audio_int16, sr, output_wav):
    # Save as WAV file
    wavfile.write(output_wav, sr, audio_int16)
        
        
    # elif task_ID =='T1L3':
        
    # elif task_ID =='T1L4':

    # elif task_ID =='T1L5':
        
    # elif task_ID =='T1L6':
        
    # elif task_ID =='T1L7':
        
    # elif task_ID =='T2L1':
        
    # elif task_ID =='T2L2':
        
    # elif task_ID =='T2L3':
        
    # elif task_ID =='T3L1':
        
    # elif task_ID =='T3L2':
