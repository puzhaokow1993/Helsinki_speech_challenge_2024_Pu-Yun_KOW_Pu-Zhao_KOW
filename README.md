# Authors, institution, location 

[Pu-Yun KOW](https://puyun321.github.io/), National Taiwan University, Taipei, Taiwan. 

[Pu-Zhao KOW](https://puzhaokow1993.github.io/homepage/index.html), National Chengchi University, Taipei, Taiwan. 

# Brief description of algorithms 

This is an audio deconvolution program, which is specially designed for the [Helsinki Speech Challenge 2024](https://blogs.helsinki.fi/helsinki-speech-challenge/). In the spirit of open science, we also provide the codes which we used to train the models. 

We use similar approach for all tasts and levels. We first transform both clean and polluted data using [fast fourier transformc (FFT)](https://numpy.org/doc/stable/reference/routines.fft.html) in `numpy`. Consequently, we train the magnitude of the transformed data using the convolutional-based auto-encoder, and we **do not** train the phase (which is highly unstable, may easily led overfitting). We multiply the trained magnitude and the original phase, and the inverse FFT outputs the denoised data. We adjust loss function for each different level by implementing some penalty terms to adjust underestimation/overestimation. 

# Installation instructions, including any requirements 

In the spirit of open science, we include all the program in this resipotory. The folder `model`, folder `output_denoise` (containing folders `Task_1_Level_1` ... `Task_3_Level_2`) and the main program `main.py` must place in a same directory.  

- Python 3.8.19 is used for the main program `main.py` with requirements in `main_requirements.txt`; but 
- Python 3.8.19 is used for the model training program (including `data_preprocessing.py`) with requirements in `training_requirements.txt`
- **Python 3.9.19** is used for the testing program `evaluate.py` with requirements in `preprocessing_evaluate_requirements.txt`. 

> [!WARNING]
> It is strongly recommended to open different (Anaconda) enviromnent to prevent the incompability of python packages. 

# Usage instructions 

The main function `main.py` can be called (via Anaconda prompt) with format 

`python main.py --input_wav_dir "path/to/files" --task_ID "xxxx"`. 

Here xxxx can be either `T1L1`, `T1L2`, `T1L3`, `T1L4`, `T1L5`, `T1L6`, `T1L7`, `T2L1`, `T2L2`, `T2L3`, `T3L1` or `T3L2`. 

> [!IMPORTANT] 
> The programs only handle 16-bit 16kHz audio files, in `.wav` format.

> [!IMPORTANT]
> One has to prepare empty folders as in the structure of the folder `output_denoise` before executing the main program `main.py`, the program unable create empty folder by itself. The structure of the folders are demonstrated as in the resipotory. One easy way is simply download the folder `model` and `output_denoise` in the resopotory and delete all the `.wav` files therein. 

> [!IMPORTANT]
> One has to change the background directory to the one which contained `main.py` (as well as the folders `model` and `output_denoise`) using the cmd command `cd` before executing the program. 

> [!NOTE] 
> (In case) If the program halt after importing `tensor flow` and `cuda`, press `enter` to continue, and the command promt will print the file names of the output denoised wav files.  

# An illustration of some example results 

We include the results (output denoised wav files) in the folder `output_denoise` above, and we also include the transcripted text `.csv` files via `evaluate.py` in the folder `results` above as well. 

> [!NOTE]
> Here the mean characater error rate (CER) are evaluated using `evaluate.py` provided by the organizer. We do not use it to train our model according to the [rules](https://blogs.helsinki.fi/helsinki-speech-challenge/rules/). 


[comment]: <> (https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)
