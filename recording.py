import numpy as np
import pandas as pd

import os, re

import librosa

def normed_hamming(M):
    h = np.hamming(M)
    return h/h.sum()

class Recording(object):

    def __init__(self,identifier,data_folder='./EmoDB'):
        self.identifier = identifier
        self.phoneme_dir = os.path.join(data_folder,'lablaut')
        self.load_phoneme_tags(os.path.join(self.phoneme_dir,identifier+'xx.lablaut'))
        self.wav_dir = os.path.join(data_folder,'wav')
        self.raw_signal, self.fs = librosa.load(os.path.join(self.wav_dir,identifier+'.wav'))
        self.mfcc_cepstrum = librosa.feature.mfcc(y=self.raw_signal,sr=self.fs,n_mfcc=20,hop_length=128)
        self.mfcc_delta1 = librosa.feature.delta(self.mfcc_cepstrum)
        self.mfcc_delta2 = librosa.feature.delta(self.mfcc_cepstrum,order=2)
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

    def pitch_features(self,raw_signal,hop_length=128):
        pitch,mag = librosa.core.piptrack(raw_signal,hop_length=hop_length)
        mean_pitch = np.mean(pitch,axis=0)
        return mean_pitch

    def mfcc_deltas(self,start,stop):
        unit = self.mfcc_cepstrum.shape[1]/(len(self.raw_signal)/self.fs)
        start = int(np.round(unit * start))
        if not start == 0:
            start = start-1
        stop = int(np.round(unit * stop))+1
        deltas = np.vstack([self.mfcc_delta1[:,start:stop],self.mfcc_delta2[:,start:stop]])
        return np.average(deltas,weights=normed_hamming(deltas.shape[1]),axis=1)

    def get_features(self,start=0,stop=None,per_phoneme=True):
        if stop == None:
            stop = len(self.raw_signal/self.fs)

        mfcc_deltas = self.mfcc_deltas(start,stop)

        stop = int(np.round(stop*self.fs))
        start = int(np.round(start*self.fs))

        mfcc = librosa.feature.mfcc(y=self.raw_signal[start:stop],sr=self.fs,n_mfcc=20,hop_length=128)
        pitch = self.pitch_features(self.raw_signal[start:stop],hop_length=128)

        features = np.vstack([mfcc,pitch])
        if per_phoneme:
            mean = np.average(features,weights=normed_hamming(mfcc.shape[1]),axis=1)
            return np.vstack([np.expand_dims(mean,-1),np.expand_dims(mfcc_deltas,-1)])
        else:
            features = np.vstack([features,self.mfcc_delta1,self.mfcc_delta2])
            return np.mean(features,axis=1)
