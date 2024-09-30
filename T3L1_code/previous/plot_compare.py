# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 13:32:04 2024

@author: steve
"""

import numpy as np

data_folder='Task_3_Level_1'
predicted_phase = np.load(r'D:\important\Hensinki_Speech_Challenge_2024\my_project\dataset\%s\pred_phase.npy'%data_folder)
predicted_phase2 = np.load(r'D:\important\Hensinki_Speech_Challenge_2024\my_project\dataset\%s\pred_phase2.npy'%data_folder)

predicted_phase_ = np.arctan2(np.sin(predicted_phase), np.cos(predicted_phase))
predicted_phase2_ = np.arctan2(np.sin(predicted_phase2), np.cos(predicted_phase2))


#%%
import matplotlib.pyplot as plt

sample = 200


plt.figure()
plt.plot(predicted_phase_ [sample,6900:7000], label='predicted_phase', color='blue')
plt.plot(predicted_phase2_[sample,6900:7000], label='predicted_phase2', color='red')
plt.show()