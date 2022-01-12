import streamlit as st
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from PIL import Image





lg_model=pickle.load(open('lg_model.pkl','rb'))
rf_model=pickle.load(open('rf_model.pkl','rb'))
dt_model=pickle.load(open('dt_model.pkl','rb'))

def classify(pred):
    if pred==0:
             return "skipped the song"
    else:
             return "song is not skipped"
         

def main():
    #img = Image.open('spotify.png')
    #img= img.resize((110,110))
    #st.image(img,use_column_width=False)
    
 
    html_temp = """
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Spotify Music Prediction</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    activities=['RandomForest','Logistic Regression','Decision Tree']
    option=st.sidebar.selectbox('Which model would you like to use?',activities)
    st.subheader(option)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        track_id = st.text_input('track_id',value=0)
        session_position = st.number_input('session_position', value=0)
        session_length = st.number_input('session_length',value=0)
        context_type = st.number_input('context_type', value=0)
        context_switch = st.number_input('context_switch',value=0)
        date = st.number_input('date', value=0)
        premium = st.number_input('premium',value=0)
    with col2:
        mode = st.number_input('mode',  value=0)
        energy = st.number_input('energy',  value=0)
        long_pause_before_play=st.number_input('long_pause_before_play',value=0)
        short_pause_before_play=st.number_input('short_pause_before_play',value=0)
        us_popularity_estimate=st.number_input("Enter us_popularity_estimate",value=0)
        tempo=st.number_input("Enter tempo",value=0)
        beat_strength=st.number_input("Enter beat_strength",value=0)
    with col3:
        loudness=st.number_input("Enter loudness",value=0)
        time_signature=st.number_input("Enter time_signature",value=0)
        dyn_range_mean=st.number_input("Enter dyn_range_mean",value=0)
        bounciness=st.number_input("Enter bounciness",value=0)
        hour_of_day=st.number_input("Enter hour_of_day",value=0)
        duration=st.number_input("Enter duration",value=0)
        valence=st.number_input('valence',value=0)
   

    
    result=" "
    inputs=[[track_id,session_position,session_length,context_type,context_switch,date,premium,mode,energy,long_pause_before_play,short_pause_before_play,us_popularity_estimate,tempo,beat_strength,loudness,time_signature,dyn_range_mean,bounciness,hour_of_day,duration,valence]]
    if st.button('submit'):
        if option=='RandomForest':
            st.success(classify(rf_model.predict(inputs)))
        elif  option=='Logistic Regression':
            st.success(classify(lg_model.predict(inputs)))
        else:
           st.success(classify(dt_model.predict(inputs)))
       

if __name__=='__main__':
    main()
