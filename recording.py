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
        self.compute_mfcc()

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

    def compute_mfcc(self):
        self.mfcc_cepstrum = librosa.feature.mfcc(y=self.raw_signal,sr=self.fs,hop_length=128,n_mfcc=30)

    def get_mfcc(self,start=0,stop=None):
        start = np.round(len(self.raw_signal)/self.mfcc_cepstrum.shape[1] * start)
        stop = np.round(len(self.raw_signal)/self.mfcc_cepstrum.shape[1] * stop)
        return self.mfcc_cepstrum[:,int(start):int(stop)]
