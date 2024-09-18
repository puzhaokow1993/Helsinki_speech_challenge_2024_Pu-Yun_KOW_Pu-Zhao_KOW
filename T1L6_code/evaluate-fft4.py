# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 19:18:23 2024

@author: steve
"""

#%% read prediction result

import numpy as np

data_folder='Task_1_Level_6'
path = r'D:\important\Hensinki_Speech_Challenge_2024\my_project\dataset\%s'%data_folder

model_name = 'ConvAE'
pred_data = np.load(r'%s\%s\pred_data-fft4.npy' % (path, model_name)).astype(np.int16)

#%% read clean and noisy file into int

import librosa
import os
import jiwer
import pandas as pd

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
    
    return audio_int

noisy_dir=r'D:\important\Hensinki_Speech_Challenge_2024\my_project\raw_dataset\%s\Recorded'%data_folder
noisy_audio_int=sampling(noisy_dir)
clean_dir=r'D:\important\Hensinki_Speech_Challenge_2024\my_project\raw_dataset\%s\Clean'%data_folder
clean_audio_int=sampling(clean_dir)

#%% data processing function

def replace_dashes(text):
    # Replace dashes '-' with spaces ' ', avoids some amgiguities.
    return text.replace("-", " ")

def replace_z(text):
    # Replace z with s, avoids confusion between british and american english
    return text.replace("z","s")

def normalize_us_spelling(text):
    # Maps some common words in the dataset from british to american
    spelling_corrections = {
    "behaviour": "behavior",
    "colour": "color",
    "favour": "favor",
    "flavour": "flavor",
    "honour": "honor",
    "humour": "humor",
    "labour": "labor",
    "neighbour": "neighbor",
    "odour": "odor",
    "savour": "savor",
    "armour": "armor",
    "clamour": "clamor",
    "enamoured": "enamored",
    "favourable": "favorable",
    "favourite": "favorite",
    "glamour": "glamor",
    "rumour": "rumor",
    "valour": "valor",
    "vigour": "vigor",
    "harbour": "harbor",
    "mould": "mold",
    "plough": "plow",
    "saviour": "savior",
    "splendour": "splendor",
    "tumour": "tumor",
    "theatre": "theater",
    "centre": "center",
    "fibre": "fiber",
    "litre": "liter",
    "metre": "meter",
    "labour": "labor",
    "labourer": "laborer",
    "kilometre": "kilometer"
}

    for british, american in spelling_corrections.items():
        text = text.replace(british, american)
    return text


def calculate_metrics(original_text, transcribed_text, transformation):
    # Apply transformations
    transformed_original = transformation(original_text)
    transformed_transcribed = transformation(transcribed_text)

    # Ensure non-empty input for calculations. If empty input, set to max error.
    if not transformed_original.strip() or not transformed_transcribed.strip():
        print(f"Empty text after transformation: Original text - {original_text}, Transcribed text - {transcribed_text}")
        return {
            "WER": 1.0,
            "CER": 1.0,
            "MER": 1.0,
            "WIL": 1.0,
            "WIP": 0.0
        }

    try:
        measures = jiwer.compute_measures(
            transformed_original, 
            transformed_transcribed,
        )
        # Additionally for the calculation of CER, remove all spaces
        cer = jiwer.cer(transformed_original.replace(" ", ""), transformed_transcribed.replace(" ",""))

        return {
            "WER": measures["wer"],
            "CER": cer,
            "MER": measures["mer"],
            "WIL": measures["wil"],
            "WIP": measures["wip"]
        }

    # Error handling
    except ValueError as e:
        print(f"Error calculating metrics: {e}")
        print(f"Transformed original text: {transformed_original}")
        print(f"Transformed transcribed text: {transformed_transcribed}")
        return None

transformation = jiwer.Compose([
    jiwer.ToLowerCase(),
    normalize_us_spelling,
    jiwer.ExpandCommonEnglishContractions(),
    replace_dashes,
    replace_z,
    jiwer.RemoveMultipleSpaces(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveWhiteSpace(replace_by_space=True),
    jiwer.Strip(),
])

#%% read deepspeech model 

from deepspeech import Model

model_path=r'D:\important\Hensinki_Speech_Challenge_2024\Evaluate_via_deepspeech\deepspeech-0.9.3-models.pbmm'
model = Model(model_path)


#%% read file name

clean_dir=r'D:\important\Hensinki_Speech_Challenge_2024\my_project\raw_dataset\%s\Clean'%data_folder
audio_files = [f for f in os.listdir(clean_dir) if f.endswith('.wav')]

#%% evaluate the performance and save result
raw_path = r'D:\important\Hensinki_Speech_Challenge_2024\my_project\raw_dataset\%s'%data_folder
real_words= pd.read_csv(r"%s\%s_text_samples.txt"%(raw_path,data_folder), delimiter='\t',header=None)
indices = np.load(r'%s\indices.npy' %(path))

full_result = []
for i in range(len(indices)):
    print(i)
    
    original_text = real_words.iloc[int(indices[i,2]),1]
    transcribed_text = model.stt(pred_data[i,:len(noisy_audio_int[i])])
    print(original_text, transcribed_text)
    
    metrics = calculate_metrics(original_text, transcribed_text, transformation)
    result = {'Filename': audio_files[int(indices[i,2])], 'Original Text': original_text, 'Transcribed Text': transcribed_text}
    if metrics:
        result.update(metrics)

    full_result.append(result)

#%%
df = pd.DataFrame(full_result)
df.to_csv(r"D:\important\Hensinki_Speech_Challenge_2024\my_project\result\%s\%s_resultt-fft4.csv"%(data_folder,model_name), index=False)




