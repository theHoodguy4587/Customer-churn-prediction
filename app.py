import streamlit as st
import pandas as pd
import joblib

model=joblib.load('logistic_regression_model.pkl')
scaler=joblib.load('scaler.pkl')
feature_columns=joblib.load('feature_names.pkl')

