import streamlit as st
import pandas as pd
import numpy as np
import base64
import pickle
import io
from scipy.io import wavfile
from scripts.test_model import perform_predictions
from pydub import AudioSegment
import joblib

def create_audio_player(audio_data, sample_rate):
    virtualfile = io.BytesIO()
    wavfile.write(virtualfile, rate=sample_rate, data=audio_data)
    return virtualfile

def main():
    st.title("Amharic Speech To Text Dashboard")

    st.sidebar.write("Navigation")
    app_mode = st.sidebar.selectbox("Choose Here", ("Home", "Model Performance", "Test Model"))
    if app_mode == 'Home':
        st.write('''
        ## Introduction
        Speech recognition technology allows for hands-free control of smartphones, speakers, and even vehicles in a 
        wide variety of languages. Companies have moved towards the goal of enabling machines to understand and respond 
        to more and more of our verbalized commands. There are many matured speech recognition systems available, 
        such as Google Assistant, Amazon Alexa, and Apple’s Siri. However, all of those voice assistants work for 
        limited languages only.
        
        The World Food Program wants to deploy an intelligent form that collects nutritional information of food bought 
        and sold at markets in two different countries in Africa - Ethiopia and Kenya. The design of this intelligent 
        form requires selected people to install an app on their mobile phone, and whenever they buy food, they use 
        their voice to activate the app to register the list of items they just bought in their own language. The 
        intelligent systems in the app are expected to live to transcribe the speech-to-text and organize the information 
        in an easy-to-process way in a database.

        Our responsibility was to build a deep learning model that is capable of transcribing a speech to text in the 
        Amharic language. The model we produce will be accurate and is robust against background noise.''')

    elif app_mode == 'Model Performance':
        st.write('''
        ## Here are a few audio Samples
        ''')
        audios, predictions, transcripts = perform_predictions('./data/wav/')
        st.subheader('Sample 1')
        st.audio(create_audio_player(audios[0], 44100))
        st.write('Prediction: '+ predictions[0])
        st.write('Actual: '+ transcripts[0])
        
        st.subheader('Sample 2')
        st.audio(create_audio_player(audios[1], 44100))
        st.write('Prediction: '+ predictions[1])
        st.write('Actual: '+ transcripts[1])

        st.subheader('Sample 3')
        st.audio(create_audio_player(audios[2], 44100))
        st.write('Prediction: '+ predictions[2])
        st.write('Actual: '+ transcripts[2])

    elif app_mode == 'Test Model':

        st.subheader('Upload the audio you want to perform predictions on')

        wav_file =  st.file_uploader("Upload Data", type = ['wav'])
        if wav_file is not None:
            st.write('Prediction: ')
            
            #file_var = AudioSegment.from_ogg(wav_file) 
            #file_var.export('./data/pred/filename.wav', format='wav')

            with open('./data/pred/tr_10001_tr097083.wav', mode='wb') as f:
                f.write(wav_file.getbuffer())

            audios, predictions, transcripts = perform_predictions('./data/pred/')
            st.subheader('Your Audio')
            st.audio(create_audio_player(audios[0], 44100))
            st.write('Prediction: '+ predictions[0])
    
if __name__ == "__main__":
    main()