# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 15:24:26 2024

@author: steve
"""

import numpy as np

data_folder='Task_1_Level_4'

output_data = np.load(r'D:\important\Hensinki_Speech_Challenge_2024\my_project\dataset\%s\new_output.npy'%data_folder)
pred_data = np.load(r'D:\important\Hensinki_Speech_Challenge_2024\my_project\dataset\%s\ConvAE\pred_data-magnitude_phase.npy'%data_folder)
pred_data2 = np.load(r'D:\important\Hensinki_Speech_Challenge_2024\my_project\dataset\%s\ConvAE\pred_data-magnitude_phase2.npy'%data_folder)

#%%
import matplotlib.pyplot as plt

sample = 1

plt.figure()
plt.plot(output_data[sample,:], label='output', color='blue')
plt.plot(pred_data[sample,:], label='predicted', color='red')
plt.show()

plt.figure()
plt.plot(output_data[sample,:], label='output', color='blue')
plt.plot(pred_data2[sample,:], label='predicted', color='red')
plt.show()