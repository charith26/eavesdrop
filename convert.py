import os
import matplotlib.pyplot as plt

import librosa
import librosa.display
import numpy as np
import pandas as pd

import IPython.display as ipd
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

audio_fpath = "dst/"
out_path = "img/"
audio_clips = os.listdir(audio_fpath)

for f in list(audio_clips):
    if f.endswith(".wav"):
        x, sr = librosa.load(audio_fpath+f, sr=44100)
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
        fig.savefig(out_path+(os.path.splitext(f)[0])+".png")
