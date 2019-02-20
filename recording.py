import numpy as np
import pandas as pd

import os, re

import librosa

class Recording(object):

    def __init__(self,identifier,data_folder='./EmoDB'):
        self.identifier = identifier
        self.phoneme_dir = os.path.join(data_folder,'lablaut')
        self.load_phoneme_tags(os.path.join(self.phoneme_dir,identifier+'xx.lablaut'))
        self.wav_dir = os.path.join(data_folder,'wav')
        self.raw_signal, self.fs = librosa.load(os.path.join(self.wav_dir,identifier+'.wav'))
        self.compute_features()
        self.df_tags.at[len(self.df_tags)-1,'t_stop'] = len(self.raw_signal)/self.fs

    def read_lines_from_file(self,filepath):
        lines = open(filepath,'r').readlines()
        for i in range(len(lines)):
            if lines[i].strip() == '#':
                break
        return lines[i+1:]

    def extract_phoneme_tags(self, s):
        aux1 = re.findall(r'[\+\-][a-z]+',s)
        for a in aux1:
            s = s.replace(a,'')
        s = re.sub(r'[\,\(\)\[\]]','',s)
        split = s.split()
        split += [''] * (2 - len(split)) # pad to len == 3
        return split[0], ','.join(aux1), split[1]

    def load_phoneme_tags(self,filepath):
        df = pd.DataFrame(columns=['t_start','t_stop','phoneme','auxiliary1','auxiliary2'])
        lines = self.read_lines_from_file(filepath)
        for i,line in enumerate(lines):
            try:
                phoneme, auxiliary1, auxiliary2 = self.extract_phoneme_tags(line[18:])
            except:
                print('Could not extract phoneme representation: ')
                print(self.extract_phoneme_tags(line[18:]))
            if i < len(lines)-1:
                t_stop = float(lines[i+1].split()[0])
            else:
                t_stop = -1
            df = df.append({
                't_start': float(line.split()[0]),
                't_stop': t_stop,
                'phoneme': phoneme.strip(),
                'auxiliary1': auxiliary1.strip(),
                'auxiliary2': auxiliary2.strip(),
            }, ignore_index=True)
        self.df_tags = df

    def mean_pitch(self,raw_signal,hop_length=128):
        pitch,mag = librosa.core.piptrack(raw_signal,hop_length=hop_length)
        mag = (mag+0.000001).T/np.sum((mag+0.000001),axis=1)
        return np.average(pitch,weights=mag.T,axis=0)

    def compute_features(self,hop_length=128):
        mfcc =  librosa.feature.mfcc(y=self.raw_signal,sr=self.fs,hop_length=hop_length,n_mfcc=20, dct_type=3)
        pitch = self.mean_pitch(self.raw_signal,hop_length)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        self.features = np.vstack([mfcc,mfcc_delta,mfcc_delta2,pitch])

    def get_features(self,start=0,stop=None):
        if stop == None:
            stop = len(self.features)
        start = np.round(len(self.raw_signal)/self.features.shape[1] * start)
        stop = np.round(len(self.raw_signal)/self.features.shape[1] * stop)
        return self.features[:,int(start):int(stop)]
