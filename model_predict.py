from attr import s
import librosa
import pandas as pd
import numpy as np
import os

# Preprocessing
from sklearn.preprocessing import StandardScaler

#Keras
import keras

import warnings
warnings.filterwarnings('ignore')

genres = np.array('blues classical country disco hiphop jazz metal pop reggae rock'.split())

def main():
    for filename in os.listdir(f'./songs'):
        y, sr = librosa.load(f"./songs/{filename}", mono=True, duration=30)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rmse = librosa.feature.rms(y=y)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        row = [np.mean(chroma_stft), np.mean(rmse), np.mean(spec_cent), np.mean(spec_bw), np.mean(rolloff), np.mean(zcr)]
        for e in mfcc:
            row.append(np.mean(e))
    X = (np.array(row))[np.newaxis, :]
    model = keras.models.load_model('model_weights')
    predictions = np.squeeze(model.predict(X))

    model_prediction = np.argmax(predictions)
    return genres[model_prediction]

if __name__ == "__main__":
    print(main())

