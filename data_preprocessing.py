# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 20:29:45 2024

@author: steve
"""

import argparse
import librosa
import numpy as np
import os
import jiwer
import pandas as pd

#%%
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
    
    return audio_files,audio_int

def fill_zeros(list1, list2):
    # Find the maximum length among all sublists in both lists
    max_len = max(
        max(len(sublist) for sublist in list1),
        max(len(sublist) for sublist in list2)
    )
    
    # Fill each sublist in list1 with zeros to match the maximum length
    list1 = [np.pad(sublist, (0, max_len - len(sublist)), 'constant') for sublist in list1]
    
    # Fill each sublist in list2 with zeros to match the maximum length
    list2 = [np.pad(sublist, (0, max_len - len(sublist)), 'constant') for sublist in list2]
    
    return list1, list2



def split_data(audio_int1, audio_int2):
    splitted_audio_int1 = []
    splitted_audio_int2 = []
    chunk_length = 16000  # 1 second

    def split_chunks(data):
        chunks = []
        num_full_chunks = len(data) // chunk_length
        remainder = len(data) % chunk_length

        # Create full chunks
        for i in range(num_full_chunks):
            chunks.append(data[i * chunk_length:(i + 1) * chunk_length])

        # Handle the remaining data
        if remainder > 0:
            last_chunk = np.zeros(chunk_length, dtype=data.dtype)
            last_chunk[:remainder] = data[num_full_chunks * chunk_length:]
            chunks.append(last_chunk)

        return chunks
    
    # splitted_audio1=[]; splitted_audio2=[]
    for index in range(len(audio_int1)):
        audio1=[]; audio2=[]
        # Split both audio inputs into chunks
        audio1.append(split_chunks(audio_int1[index]))
        audio2.append(split_chunks(audio_int2[index]))
    
        # Determine the minimum number of full chunks
        print(len(audio1[0]),len(audio2[0]))
        min_num_chunks = min(len(audio1[0]), len(audio2[0])) 
    
        # Truncate the longer list to match the length of the shorter one
        audio1 = audio1[0][:min_num_chunks]
        audio2 = audio2[0][:min_num_chunks]

        splitted_audio_int1.append(audio1)
        splitted_audio_int2.append(audio2)
    return splitted_audio_int1, splitted_audio_int2


def adjust_signal(signal_2, signal_1):
    # Compute the cross-correlation between the two signals
    correlation = np.correlate(signal_1 - np.mean(signal_1), signal_2 - np.mean(signal_2), mode='full')
    
    # Find the lag (shift) corresponding to the maximum cross-correlation
    lag = np.argmax(correlation) - (len(signal_1) - 1)
    
    # Shift the second signal based on the lag
    if lag > 0:
        aligned_signal_2 = np.roll(signal_2, lag)
        aligned_signal_2[:lag] = 0  # Set shifted portion to 0 to avoid misalignment
        aligned_signal_1 = signal_1
    else:
        aligned_signal_2 = np.roll(signal_2, lag)
        aligned_signal_2[lag:] = 0  # Set shifted portion to 0 for negative lag
        aligned_signal_1 = signal_1
    
    # Trim both signals to the same length after shifting
    min_length = min(len(aligned_signal_2), len(aligned_signal_1))
    aligned_signal_1 = aligned_signal_1[:min_length]
    aligned_signal_2 = aligned_signal_2[:min_length]
    
    return aligned_signal_2, aligned_signal_1

def get_intersect(noise_list,clean_list):
    list1 = noise_list; list2= clean_list
    # Remove the word 'recorded' from each string
    list1 = [name.replace("recorded_", "") for name in list1]
    list2 = [name.replace("clean_", "") for name in list2]

    if len(list2)<len(list1):
        # Convert list2 to a set for faster lookup
        set_list2 = set(list2)
        
        # Get intersection with indices from list1 and list2
        intersection_with_indices = [(item, idx1, list2.index(item)) 
                                     for idx1, item in enumerate(list1) if item in set_list2]
    else:
        set_list1 = set(list1)
        
        # Get intersection with indices from list1 and list2
        intersection_with_indices = [(item, list1.index(item), idx2) 
                                     for idx2, item in enumerate(list2) if item in set_list1]
    return intersection_with_indices 

#%%
raw_folders=['Task_1_Level_1','Task_1_Level_2','Task_1_Level_3','Task_1_Level_4','Task_1_Level_5','Task_1_Level_6',
          'Task_1_Level_7','Task_2_Level_1','Task_2_Level_2','Task_2_Level_3','Task_3_Level_1','Task_3_Level_2']

for i in range(1,len(raw_folders)):
    # Directory and file processing
    noisy_dir=r'D:\important\Hensinki_Speech_Challenge_2024\my_project\raw_dataset\%s\Recorded'%raw_folders[i]
    noisy_filename,noisy_audio_int=sampling(noisy_dir)
            
    clean_dir=r'D:\important\Hensinki_Speech_Challenge_2024\my_project\raw_dataset\%s\Clean'%raw_folders[i]
    clean_filename,clean_audio_int=sampling(clean_dir)
    
    indices=get_intersect(noisy_filename,clean_filename)
    
    new_noisy_audio_int=[]; new_clean_audio_int=[]
    
    for j in range(len(indices)):
        print(j)
        noisy_aligned, clean_aligned = adjust_signal(noisy_audio_int[indices[j][1]], clean_audio_int[indices[j][2]])
        new_noisy_audio_int.append(noisy_aligned)
        new_clean_audio_int.append(clean_aligned)
    
    noisy_audio_int,clean_audio_int=fill_zeros(new_noisy_audio_int,new_clean_audio_int)
    indices = np.array(indices)
    
    
    np.save(r'D:\important\Hensinki_Speech_Challenge_2024\my_project\dataset\%s\indices.npy'%raw_folders[i], indices)
    np.save(r'D:\important\Hensinki_Speech_Challenge_2024\my_project\dataset\%s\new_input.npy'%raw_folders[i], noisy_audio_int)
    np.save(r'D:\important\Hensinki_Speech_Challenge_2024\my_project\dataset\%s\new_output.npy'%raw_folders[i], clean_audio_int)

