import sounddevice as sd
from scipy.io.wavfile import write
import pygame
from pygame.locals import *
import os
import matplotlib.pyplot as plt

import pydub
from pydub import AudioSegment
from pydub import AudioSegment
from pydub.utils import make_chunks

import librosa
import librosa.display
import numpy as np
import pandas as pd

import sys

from google.cloud import automl_v1beta1
from google.cloud.automl_v1beta1.proto import service_pb2

import IPython.display as ipd
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from time import sleep
from os import system, name

file_name = "key.wav"
project_id = "your google project id"
model_id = "your model id"

def main():
    while 1:
        key = recordAudio()
        trimAudio()
        createImage()
        with open("key.png", 'rb') as f:
            content = f.read()
        print("Pressed Key: "+key)
        print("Prediction : "+get_prediction(content, project_id, model_id))
        sleep(2)

def recordAudio():
    clear()
    key = ""
    pygame.init()

    fs = 44100
    seconds = 5

    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    print("Press a key from a to h:")

    while 1:
            event = pygame.event.poll()
            if event.type == pygame.KEYUP:
                sd.stop()
                write(file_name, fs, myrecording)
                key = pygame.key.name(event.key)
                break
            else:
                continue
    pygame.display.quit()
    pygame.quit()
    return key

def createImage():
    x, sr = librosa.load(file_name, sr=44100)
    window_size = 1024
    window = np.hanning(window_size)
    stft  = librosa.core.spectrum.stft(x, n_fft=window_size, hop_length=512, window=window)
    out = 2 * np.abs(stft) / np.sum(window)
    fig = plt.Figure()
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    p = librosa.display.specshow(librosa.amplitude_to_db(out, ref=np.max), ax=ax, y_axis='log', x_axis='time')
    fig.savefig((os.path.splitext(file_name)[0])+".png")
    return file_name

def trimAudio():
    sound = AudioSegment.from_wav(file_name)

    start_trim = pydub.silence.detect_nonsilent(sound, min_silence_len=100, silence_thresh=-80)

    duration = len(sound)
    trimmed_sound = sound[start_trim[0][0]:start_trim[0][1]]
    trimmed_sound.export( file_name, format="wav")

def get_prediction(content, project_id, model_id):
  prediction_client = automl_v1beta1.PredictionServiceClient()

  name = 'projects/{}/locations/us-central1/models/{}'.format(project_id, model_id)
  payload = {'image': {'image_bytes': content }}
  params = {}
  request = prediction_client.predict(name, payload, params)
  return request.payload[0].display_name  # waits till request is returned

def clear():
    if name == 'nt':
        _ = system('cls')
    else:
        _ = system('clear')

if __name__ == '__main__':
    main()
