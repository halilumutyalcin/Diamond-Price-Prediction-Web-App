from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np


def predict_price(model, df):
    predictions_data = predict_model(estimator=model, data=df)
    return predictions_data["prediction_label"].iloc[0]


model = load_model('best_model_api')

st.title('Diamond Price Predictor Web App')
st.write('This is a web app to predict the price of your diamond based on\
         several features that you can see in the sidebar. Please adjust the\
         value of each feature. After that, click on the Predict button at the bottom to\
         see the prediction of the diamond.')

carat = st.sidebar.slider(label='Carat', min_value=0.0,
                          max_value=5.0,
                          value=2.5,
                          step=0.1)

# cut = st.sidebar.slider(label='Cut', min_value=0,
#                         max_value=4,
#                         value=2,
#                         step=1)
# cut =
cut_dict = {'Fair': 0, 'Good': 1, 'Ideal': 2, 'Premium': 3, 'Very Good': 4}
cut = cut_dict[st.sidebar.selectbox("Cut",('Ideal','Premium','Good','Very Good','Fair'))]

color_dict = {'D': 0, 'E': 1, 'F': 2, 'G': 3, 'H': 4, 'I': 5, 'J': 6}
color = color_dict[st.sidebar.selectbox("Color",('D', 'E', 'F', 'G', 'I','J'))]



clarity_dict = {'I1': 0,
  'IF': 1,
  'SI1': 2,
  'SI2': 3,
  'VS1': 4,
  'VS2': 5,
  'VVS1': 6,
  'VVS2': 7}
clarity = clarity_dict[st.sidebar.selectbox("Clarity",clarity_dict.keys())]


depth = st.sidebar.slider(label='Depth', min_value=30.0,
                          max_value=80.0,
                          value=40.0,
                          step=0.1)

table = st.sidebar.slider(label='Table', min_value=40.0,
                          max_value=100.0,
                          value=70.0,
                          step=0.1)

X = st.sidebar.slider(label='X', min_value=0.00,
                      max_value=11.00,
                      value=5.00,
                      step=0.01)

Y = st.sidebar.slider(label='Y', min_value=0.00,
                      max_value=60.00,
                      value=30.00,
                      step=0.01)

Z = st.sidebar.slider(label='Z', min_value=2.00,
                      max_value=40.00,
                      value=20.00,
                      step=0.01)

features = {'carat': carat, 'cut': cut, 'color': color,
            'clarity': clarity,
            'table': table, 'x': X, 'y': Y,
            'z': Z, 'depth': depth
            }

features_df = pd.DataFrame([features])

st.table(features_df)

if st.button('Predict'):
    prediction = predict_price(model, features_df)

    st.write(' Based on feature values, your diamond price is ' + str(round(prediction,2)) + "$")
