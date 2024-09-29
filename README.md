# Authors, institution, location 

[Pu-Yun KOW](https://puyun321.github.io/), National Taiwan University, Taipei, Taiwan. 

[Pu-Zhao KOW](https://puzhaokow1993.github.io/homepage/index.html), National Chengchi University, Taipei, Taiwan. 

# Brief description of algorithms 

This is an audio deconvolution program, which is specially designed for the [Helsinki Speech Challenge 2024](https://blogs.helsinki.fi/helsinki-speech-challenge/). In the spirit of open science, we also provide the codes which we used to train the models. 

**Task 1 (T1L1 - T1L7)** 
> We first transform both clean and polluted data using [fast fourier transformc (FFT)](https://docs.scipy.org/doc/scipy/tutorial/fft.html). Consequently, we train the magnitude of the transformed data using the convolutional-based auto-encoder, and we **do not** train the phase (which is highly unstable, may easily led overfitting). We multiply the trained magnitude and the original phase, and the inverse FFT outputs the denoised data. We adjust loss function for each different level by implementing some penalty terms to adjust underestimation/overestimation. 

- [ ] Pu-Zhao comments: Task 2 is in progress. 



# Installation instructions, including any requirements 

In the spirit of open science, we include all the program in this resipotory. 

- Python 3.8.19 is used for the main program `main.py` with requirements in `main_requirements.txt`; but 
- Python 3.8.19 is used for the model training program (including `data_preprocessing.py`) with requirements in `training_requirements.txt`
- **Python 3.9.19** is used for the testing program `evaluate.py` with requirements in `preprocessing_evaluate_requirements.txt`. 


> [!WARNING]
> It is strongly recommended to open different enviromnent to prevent the incompability of python packages. 

- [ ] Pu-Zhao comments: make sure the program can be run by just unzip the zip file. 

# Usage instructions 

- [ ] Pu-Zhao comments: TBA 

> [!IMPORTANT] 
> The programs only handle 16-bit 16kHz audio files, in `.wav` format.

> [!IMPORTANT] 
> The name of the directories/folders must not be changed. 

# An illustration of some example results 

- [ ] Pu-Zhao comments: TBA (put some audio files for demonstration, and also show the mean CER)

> [!NOTE]
> Here the mean characater error rate (CER) are evaluated using `evaluate.py` provided by the organizer. We do not use it to train our model according to the [rules](https://blogs.helsinki.fi/helsinki-speech-challenge/rules/). 



[comment]: <> (https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)
