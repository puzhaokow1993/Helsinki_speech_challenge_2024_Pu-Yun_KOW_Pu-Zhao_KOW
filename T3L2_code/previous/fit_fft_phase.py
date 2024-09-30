# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 16:19:18 2024

@author: Steve
"""

import numpy as np

data_folder='Task_3_Level_2'
input_data = np.load(r'D:\important\Hensinki_Speech_Challenge_2024\my_project\dataset\%s\unwrapped_input_fft_phase.npy'%data_folder)
output_data = np.load(r'D:\important\Hensinki_Speech_Challenge_2024\my_project\dataset\%s\unwrapped_output_fft_phase.npy'%data_folder)

output_difference = output_data - input_data

#%%
from sklearn.decomposition import PCA
# Reduce dimensionality to avoid memory issues
pca = PCA(n_components=50)  # Adjust the number of components as needed
X_reduced = pca.fit_transform(input_data)

#%%
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)  # Create polynomial features (degree=2 for quadratic)
X_poly = poly.fit_transform(X_reduced)  # Transform the input data

# Create a linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_poly, output_difference)

joblib.dump(pca, 'D:\important\Hensinki_Speech_Challenge_2024\my_project\model\%s\pca_model.pkl'%data_folder)  # Save the PCA transformer
joblib.dump(poly, 'D:\important\Hensinki_Speech_Challenge_2024\my_project\model\%s\poly_transformer.pkl'%data_folder)
joblib.dump(model, 'D:\important\Hensinki_Speech_Challenge_2024\my_project\model\%s\polynomial_regression_model.pkl'%data_folder)  # Save the trained regression model

#%%
# Later, you can load and use them like this:
pca_loaded = joblib.load('D:\important\Hensinki_Speech_Challenge_2024\my_project\model\%s\pca_model.pkl'%data_folder)
poly_loaded = joblib.load('D:\important\Hensinki_Speech_Challenge_2024\my_project\model\%s\poly_transformer.pkl'%data_folder)
model_loaded = joblib.load('D:\important\Hensinki_Speech_Challenge_2024\my_project\model\%s\polynomial_regression_model.pkl'%data_folder)

# To make predictions using the saved models:
X_reduced = pca_loaded.transform(input_data)  # Apply PCA to new data
X_poly = poly_loaded.transform(X_reduced)
pred_difference  = model_loaded.predict(X_poly)  # Make predictions
pred_data = input_data + pred_difference

np.save(r'D:\important\Hensinki_Speech_Challenge_2024\my_project\dataset\%s\pred_phase.npy'%data_folder, pred_data)

#%%
import matplotlib.pyplot as plt

sample = 200


plt.figure()
plt.plot(output_data[sample,:], label='output', color='blue')
plt.plot(pred_data[sample,:], label='predicted', color='red')
plt.show()

plt.figure()
plt.plot(input_data[sample,:], label='output', color='blue')
plt.plot(pred_data[sample,:], label='input', color='red')
plt.show()