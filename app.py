import pandas as pd 
import pickle as pkl
from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st 
from spModel import predict

#import model
model = pkl.load(open('model/ESPD.pkl', 'rb'))

st.header('**Sapm Detection System**')

#get input data
incoming_message = st.text_input('**Enter Your Message**')
if st.button('validate'):
    output = predict(incoming_message)
    st.markdown(f"**Input Message:** {incoming_message}")
    st.markdown(f"**Predicted Output:** The input message is {output}")
