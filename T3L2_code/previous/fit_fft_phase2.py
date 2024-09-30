# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 16:19:18 2024

@author: Steve
"""

import numpy as np

data_folder='Task_3_Level_1'
input_data = np.load(r'D:\important\Hensinki_Speech_Challenge_2024\my_project\dataset\%s\unwrapped_input_fft_phase.npy'%data_folder)
output_data = np.load(r'D:\important\Hensinki_Speech_Challenge_2024\my_project\dataset\%s\unwrapped_output_fft_phase.npy'%data_folder)

# output_difference = output_data - input_data

#%%
from sklearn.decomposition import PCA
# Reduce dimensionality to avoid memory issues
pca = PCA(n_components=100)  # Adjust the number of components as needed
X_reduced = pca.fit_transform(input_data)

pca2 = PCA(n_components=100)  # Adjust the number of components as needed
y_reduced = pca2.fit_transform(output_data)

output_difference = y_reduced + X_reduced

#%%
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)  # Create polynomial features (degree=2 for quadratic)
X_poly = poly.fit_transform(X_reduced)  # Transform the input data

#%%
import joblib
from sklearn.linear_model import LinearRegression
# Create a linear regression model
model = LinearRegression()

# Train the model on the training data
# model.fit(X_poly, y_reduced)
model.fit(X_reduced, output_difference)

joblib.dump(pca, 'D:\important\Hensinki_Speech_Challenge_2024\my_project\model\Task_3_Level_1\pca_model.pkl')  # Save the PCA transformer
joblib.dump(pca2, 'D:\important\Hensinki_Speech_Challenge_2024\my_project\model\Task_3_Level_1\pca2_model.pkl')  # Save the PCA transformer
joblib.dump(poly, 'D:\important\Hensinki_Speech_Challenge_2024\my_project\model\Task_3_Level_1\poly2_transformer.pkl')
joblib.dump(model, 'D:\important\Hensinki_Speech_Challenge_2024\my_project\model\Task_3_Level_1\linear_regression_model.pkl')  # Save the trained regression model

#%%
# Later, you can load and use them like this:
pca_loaded = joblib.load('D:\important\Hensinki_Speech_Challenge_2024\my_project\model\Task_3_Level_1\pca_model.pkl')
pca2_loaded = joblib.load('D:\important\Hensinki_Speech_Challenge_2024\my_project\model\Task_3_Level_1\pca2_model.pkl')
poly_loaded = joblib.load('D:\important\Hensinki_Speech_Challenge_2024\my_project\model\Task_3_Level_1\poly2_transformer.pkl')
model_loaded = joblib.load('D:\important\Hensinki_Speech_Challenge_2024\my_project\model\Task_3_Level_1\linear_regression_model.pkl')

# To make predictions using the saved models:
X_reduced = pca_loaded.transform(input_data)  # Apply PCA to new data
X_poly = poly_loaded.transform(X_reduced)  # Transform the input data
pred_data  = model_loaded.predict(X_poly)  # Make predictions
pred_data  = pca2_loaded.inverse_transform(pred_data)  

pred_difference  = model_loaded.predict(X_poly)  # Make predictions
pred_data  = pca2_loaded.inverse_transform(pred_difference+X_reduced)  

# pred_data=np.arctan2(np.sin(pred_data), np.cos(pred_data))

#%%
np.save(r'D:\important\Hensinki_Speech_Challenge_2024\my_project\dataset\%s\pred_phase2.npy'%data_folder, pred_data)

#%% plot compare if needed
# output_data = np.arctan2(np.sin(output_data), np.cos(output_data))

import matplotlib.pyplot as plt


sample = 279
plt.figure()
plt.plot(output_data[sample,:], label='output', color='blue')
plt.plot(pred_data[sample,:], label='predicted', color='red')
plt.show()

plt.figure()
plt.plot(input_data[sample,:], label='output', color='blue')
plt.plot(pred_data[sample,:], label='input', color='red')
plt.show()