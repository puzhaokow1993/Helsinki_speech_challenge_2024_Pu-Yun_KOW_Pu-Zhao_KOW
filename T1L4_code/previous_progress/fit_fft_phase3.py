# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 16:19:18 2024

@author: Steve
"""

import numpy as np

data_folder='Task_1_Level_4'
input_data = np.load(r'D:\important\Hensinki_Speech_Challenge_2024\my_project\dataset\%s\unwrapped_input_fft_phase.npy'%data_folder)
output_data = np.load(r'D:\important\Hensinki_Speech_Challenge_2024\my_project\dataset\%s\unwrapped_output_fft_phase.npy'%data_folder)

#%%
def wrap_back(unwrapped_phase):
    wrapped_back_phase = np.arctan2(np.sin(unwrapped_phase), np.cos(unwrapped_phase))

    # wrapped_back_phase = (unwrapped_phase + np.pi) % (2 * np.pi) - np.pi
    return wrapped_back_phase

input_data = wrap_back(input_data)
output_data = wrap_back(output_data)

#%%
from sklearn.decomposition import PCA
# Reduce dimensionality to avoid memory issues
pca = PCA(n_components=611)  # Adjust the number of components as needed
X_reduced = pca.fit_transform(input_data)

pca2 = PCA(n_components=611)  # Adjust the number of components as needed
y_reduced = pca2.fit_transform(output_data)

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
# poly_loaded = joblib.load('D:\important\Hensinki_Speech_Challenge_2024\my_project\model\Task_1_Level_4\poly2_transformer.pkl')
model_loaded = joblib.load('D:\important\Hensinki_Speech_Challenge_2024\my_project\model\Task_1_Level_4\linear_regression_model.pkl')

# To make predictions using the saved models:
X_reduced = pca_loaded.transform(input_data)  # Apply PCA to new data
# X_poly = poly_loaded.transform(X_reduced)  # Transform the input data
# pred_difference  = model_loaded.predict(X_poly)  # Make predictions

pred_data  = model_loaded.predict(X_reduced)  # Make predictions
pred_data  = pca2_loaded.inverse_transform(pred_data)  

# pred_data=np.arctan2(np.sin(pred_data), np.cos(pred_data))

#%%
np.save(r'D:\important\Hensinki_Speech_Challenge_2024\my_project\dataset\%s\pred_phase2.npy'%data_folder, pred_data)

#%% plot compare if needed
# output_data = np.arctan2(np.sin(output_data), np.cos(output_data))

import matplotlib.pyplot as plt

sample = 500

# plt.figure()
# plt.plot(X_reduced[sample,:], label='output', color='blue')
# plt.plot(y_reduced[sample,:], label='predicted', color='red')
# plt.show()


plt.figure()
plt.plot(pred_data[sample,15000:15500], label='predicted', color='red')
plt.plot(output_data[sample,15000:15500], label='output', color='blue')
plt.show()

plt.figure()
plt.plot(input_data[sample,15000:15500], label='input', color='red')
plt.plot(output_data[sample,15000:15500], label='output', color='blue')
plt.show()