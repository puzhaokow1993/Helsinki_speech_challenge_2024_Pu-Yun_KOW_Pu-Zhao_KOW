# Authors, institution, location 

[Pu-Yun KOW](https://puyun321.github.io/), National Taiwan University, Taipei, Taiwan. 

[Pu-Zhao KOW](https://puzhaokow1993.github.io/homepage/index.html), National Chengchi University, Taipei, Taiwan. 

# Brief description of algorithms 

This is an audio deconvolution program, which is specially designed for the [Helsinki Speech Challenge 2024](https://blogs.helsinki.fi/helsinki-speech-challenge/). In the spirit of open science, we also provide the codes which we used to train the models. 

Filtering Level 1 to Level 3 (T1L1, T1L2, T1L3) 
> We first transform both clean and polluted data using [fast fourier transformc (FFT)](https://docs.scipy.org/doc/scipy/tutorial/fft.html). Consequently, we train the magnitude of the transformed data, and we **do not** train the phase. We multiply the trained magnitude and the original phase, and the inverse FFT outputs the denoised data. 

- [] Pu-Zhao comments: Explain how to use CNN to train the magnitude 

Filtering Level 4 to Level 7 (T1L4, T1L5, T1L6, T1L7) 
> We basically use the same ideas as above, but now we also train the phase. The problem is tricky, since the phase has period 2π. 



# Installation instructions, including any requirements 

For simplicity, we compressed all necessary files into a single zip file (including the list of requirement packages in requirement.txt). 
- Python 3.x is required for main.py
- Python 3.x is required for the model training program 

# Usage instructions 

TBA 

> [!IMPORTANT] 
> The programs only handle 16-bit 16kHz audio files, in `.wav` format.

> [!IMPORTANT] 
> The name of the directories/folders must not be changed. 

# An illustration of some example results 

TBA (put some audio files for demonstration, and also show the mean CER )

> [!NOTE]
> Here the mean characater error rate (CER) are evaluated using `evaluate.py` provided by the organizer. We do not use it to train our model according to the [rules](https://blogs.helsinki.fi/helsinki-speech-challenge/rules/). 



[comment]: <> (https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)
