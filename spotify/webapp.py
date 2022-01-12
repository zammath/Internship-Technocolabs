import streamlit as st
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


#lg_model=pickle.load(open('lg_model.pkl','rb'))
rf_model=pickle.load(open('rf_model.pkl','rb'))



col1, mid, col2 = st.columns([1,1,20])
with col1:
    st.image('spotify.png', width=90)
with col2:
    st.write("""
# Predicting Spotify Track Skips
This app predicts the **Spotify Track Skips**
""")
    st.sidebar.header('User Input Parameters')



def user_input_features():
    track_id = st.sidebar.text_input('track_id',0)
    session_position = st.sidebar.text_input('session_position', 0, 1, 1)
    session_length = st.sidebar.text_input('session_length',0,1,1)
    context_type = st.sidebar.text_input('context_type', 0.0,1,1)
    context_switch = st.sidebar.text_input('context_switch',0.0,1,1)
    #Hours = st.sidebar.text_input('Hours', 0)
    date = st.sidebar.text_input('date', 1,31)
    premium = st.sidebar.number_input('premium', 0)
    mode = st.sidebar.number_input('mode',  0, 1)
    energy = st.sidebar.number_input('energy',  0, 1, 1)
    long_pause_before_play=st.sidebar.number_input('long_pause_before_play',0)
    short_pause_before_play=st.sidebar.number_input('short_pause_before_play',0)
    #flatness=st.sidebar.number_input("Enter flatness",value=0)
    liveness=st.sidebar.number_input("Enter liveness",value=0)
    #key=st.sidebar.number_input("Enter key",value=0)
    energy=st.sidebar.number_input("Enter energy",value=0)
    us_popularity_estimate=st.sidebar.number_input("Enter us_popularity_estimate",value=0)
    tempo=st.sidebar.number_input("Enter tempo",value=0)
    beat_strength=st.sidebar.number_input("Enter beat_strength",value=0)
    loudness=st.sidebar.number_input("Enter loudness",value=0)
    #dancebility=st.sidebar.number_input("Enter dancebility",value=0)
    time_signature=st.sidebar.number_input("Enter time_signature",value=0)
    dyn_range_mean=st.sidebar.number_input("Enter dyn_range_mean",value=0)
    bounciness=st.sidebar.number_input("Enter bounciness",value=0)
    #acoustic_vector_3=st.sidebar.number_input("Enter acoustic_vector_3",value=0)
    hour_of_day=st.sidebar.number_input("Enter hour_of_day",value=0)
    #acoustic_vector_2=st.sidebar.number_input("Enter acoustic_vector_2",value=0)
    #acoustic_vector_7=st.sidebar.number_input("Enter acoustic_vector_7",value=0)
    duration=st.sidebar.number_input("Enter duration",value=0)
    #acoustic_vector_1=st.sidebar.number_input("Enter acoustic_vector_1",value=0)
    #speechiness=st.sidebar.number_input("Enter speechiness",value=0)
    valence=st.sidebar.number_input('valence',0)


    data = {'track_id':track_id,
            'session_position':session_position,
            'session_length':session_length,
            'context_type':context_type,
            'context_switch':context_switch,
            'date':date,
            'premium':premium,
            'tempo':tempo,
            'mode':mode,
            'energy':energy,
            'us_popularity_estimate':us_popularity_estimate,
            'beat_strength':beat_strength,
            'loudness':loudness,
            'time_signature':time_signature,
            'dyn_range_mean':dyn_range_mean,
            'bounciness':bounciness,
            'hour_of_day':hour_of_day,
            'duration':duration,
            'long_pause_before_play':long_pause_before_play,
            'short_pause_before_play':short_pause_before_play,
            'valence':valence,
            }


    features = pd.DataFrame(data, index=[0])
    return features

df1 = user_input_features()

st.subheader('Class labels and their corresponding index number')
st.write(df1)

col1,mid, col2 = st.columns([20,1,20])
with col1:
    st.subheader('Prediction')
    prediction = rf_model.predict(df1)
    skip = np.array(['Skipped','Not Skipped'])
    st.write(skip[prediction])
with col2:
    st.subheader('Prediction Probability')
    predict_proba=rf_model.predict_proba(df1)
    st.write(predict_proba)

st.markdown("<h3 style='text-align: center;'>Author: Zammath </h3>", unsafe_allow_html=True)