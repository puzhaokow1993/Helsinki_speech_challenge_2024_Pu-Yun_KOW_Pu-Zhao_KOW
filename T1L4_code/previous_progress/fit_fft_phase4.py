# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 16:19:18 2024

@author: Steve
"""

import numpy as np

data_folder='Task_1_Level_4'
input_phase = np.load(r'D:\important\Hensinki_Speech_Challenge_2024\my_project\dataset\%s\input_fft_phase.npy'%data_folder)
output_phase = np.load(r'D:\important\Hensinki_Speech_Challenge_2024\my_project\dataset\%s\output_fft_phase.npy'%data_folder)

#%%
lag = np.load(r'D:\important\Hensinki_Speech_Challenge_2024\my_project\dataset\%s\lag.npy'%data_folder)

def adjust_signal(signal, lag):
    
    aligned_signal = np.roll(signal, -lag)
    if lag<0:
        aligned_signal[:-lag] = 0  # Set shifted portion to 0 to avoid misalignment
    else:
        aligned_signal[-lag:] = 0  # Set shifted portion to 0 to avoid misalignment

    return aligned_signal

shift_input_phase = np.array([adjust_signal(input_phase[i,:],lag[i]) for i in range(len(input_phase))])
shift_output_phase = np.array([adjust_signal(output_phase[i,:],lag[i]) for i in range(len(output_phase))])

input_data = np.concatenate([input_phase, shift_input_phase])
output_data = np.concatenate([output_phase, shift_output_phase])

#%%
from sklearn.decomposition import PCA
# Reduce dimensionality to avoid memory issues
pca = PCA(n_components=len(input_phase))  # Adjust the number of components as needed
X_reduced = pca.fit_transform(input_phase)

pca2 = PCA(n_components=len(output_phase))  # Adjust the number of components as needed
y_reduced = pca2.fit_transform(output_phase)

X_reduced = pca.transform(input_data)  # Apply PCA to new data
y_reduced = pca2.transform(output_data)  # Apply PCA to new data

#%%
# from sklearn.preprocessing import PolynomialFeatures

# poly = PolynomialFeatures(degree=2)  # Create polynomial features (degree=2 for quadratic)
# X_poly = poly.fit_transform(X_reduced)  # Transform the input data

#%%
import joblib
from sklearn.linear_model import LinearRegression
# Create a linear regression model
model = LinearRegression()

# Train the model on the training data
# model.fit(X_poly, y_reduced)
model.fit(X_reduced, y_reduced)

joblib.dump(pca, 'D:\important\Hensinki_Speech_Challenge_2024\my_project\model\Task_1_Level_4\pca2_model.pkl')  # Save the PCA transformer
joblib.dump(pca2, 'D:\important\Hensinki_Speech_Challenge_2024\my_project\model\Task_1_Level_4\pca2-2_model.pkl')  # Save the PCA transformer
# joblib.dump(poly, 'D:\important\Hensinki_Speech_Challenge_2024\my_project\model\Task_1_Level_4\poly2_transformer.pkl')
joblib.dump(model, 'D:\important\Hensinki_Speech_Challenge_2024\my_project\model\Task_1_Level_4\linear_regression_model.pkl')  # Save the trained regression model

#%%
import joblib
from sklearn.linear_model import LinearRegression
# Later, you can load and use them like this:
pca_loaded = joblib.load('D:\important\Hensinki_Speech_Challenge_2024\my_project\model\Task_1_Level_4\pca2_model.pkl')
pca2_loaded = joblib.load('D:\important\Hensinki_Speech_Challenge_2024\my_project\model\Task_1_Level_4\pca2-2_model.pkl')
poly_loaded = joblib.load('D:\important\Hensinki_Speech_Challenge_2024\my_project\model\Task_1_Level_4\poly2_transformer.pkl')
model_loaded = joblib.load('D:\important\Hensinki_Speech_Challenge_2024\my_project\model\Task_1_Level_4\linear_regression_model.pkl')

# To make predictions using the saved models:
X_reduced = pca_loaded.transform(input_data)  # Apply PCA to new data
# X_poly = poly_loaded.transform(X_reduced)  # Transform the input data
# pred_data  = model_loaded.predict(X_poly)  # Make predictions

pred_data  = model_loaded.predict(X_reduced)  # Make predictions
pred_data  = pca2_loaded.inverse_transform(pred_data)  

# pred_data=np.arctan2(np.sin(pred_data), np.cos(pred_data))

#%%
np.save(r'D:\important\Hensinki_Speech_Challenge_2024\my_project\dataset\%s\pred_phase2.npy'%data_folder, pred_data)

#%% plot compare if needed
# output_data = np.arctan2(np.sin(output_data), np.cos(output_data))

import matplotlib.pyplot as plt

sample = 0

plt.figure()
plt.plot(X_reduced[sample,:], label='output', color='blue')
plt.plot(y_reduced[sample,:], label='predicted', color='red')
plt.show()


plt.figure()
plt.plot(pred_data[sample,15000:15500], label='predicted', color='red')
plt.plot(output_data[sample,15000:15500], label='output', color='blue')
plt.show()

plt.figure()
plt.plot(input_data[sample,15000:15500], label='input', color='red')
plt.plot(output_data[sample,15000:15500], label='output', color='blue')
plt.show()