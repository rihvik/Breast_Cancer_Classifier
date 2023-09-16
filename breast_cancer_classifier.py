# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pickle
import numpy as np
import streamlit as st

loaded_model = pickle.load(open("breast_cancer_classifier.sav",'rb'))

def breast_cancer_classifier(input_data):
    input_data_as_np_array = np.asarray(input_data)
    input_data_reshape = input_data_as_np_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshape)
    print(prediction)
    if prediction[0] == 0:
        return 'The person does not have breast cancer'

def main():
    st.title("Breast Cancer Classifier")
    mean_radius = st.text_input('mean radius')
    mean_texture = st.text_input('mean texture')
    mean_perimeter = st.text_input('mean perimeter')
    mean_area = st.text_input('mean area')
    mean_smoothness = st.text_input('mean smoothness')
    mean_compactness = st.text_input('mean compactness')
    mean_concavity = st.text_input('mean concavity')
    mean_concave_points = st.text_input('mean concave points')
    mean_symmetry = st.text_input('mean symmetry')
    mean_fractal_dimension = st.text_input('mean fractal dimension')
    radius_error = st.text_input('radius error')
    texture_error = st.text_input('texture error')
    perimeter_error = st.text_input('perimeter error')
    area_error = st.text_input('area error')
    smoothness_error = st.text_input('smoothness error')
    compactness_error = st.text_input('compactness error')
    concavity_error = st.text_input('concavity error')
    concave_points_error = st.text_input('concave points error')
    symmetry_error = st.text_input('symmetry error')
    fractal_dimension_error = st.text_input('fractal dimension error')
    worst_radius = st.text_input('worst radius')
    worst_texture = st.text_input('worst texture')
    worst_perimeter = st.text_input('worst perimeter')
    worst_area = st.text_input('worst area')
    worst_smoothness = st.text_input('worst smoothness')
    worst_compactness = st.text_input('worst compactness')
    worst_concavity = st.text_input('worst concavity')
    worst_concave_points = st.text_input('worst concave points')
    worst_symmetry = st.text_input('worst symmetry')
    worst_fractal_dimension = st.text_input('worst fractal dimension')
    
    # available for prediction
    diagnosis = ''
    if st.button('Breast Cancer Result:'):
        # Concatenate feature names and their respective values
        features_values = [mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness,
                           mean_compactness, mean_concavity, mean_concave_points, mean_symmetry,
                           mean_fractal_dimension, radius_error, texture_error, perimeter_error,
                           area_error, smoothness_error, compactness_error, concavity_error,
                           concave_points_error, symmetry_error, fractal_dimension_error,
                           worst_radius, worst_texture, worst_perimeter, worst_area,
                           worst_smoothness, worst_compactness, worst_concavity,
                           worst_concave_points, worst_symmetry, worst_fractal_dimension]

        # Call the breast_cancer_classifier function with the concatenated feature values
        diagnosis = breast_cancer_classifier(features_values)

    st.success(diagnosis)

if __name__ == '__main__':
    main()
