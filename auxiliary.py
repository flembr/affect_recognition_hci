import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm

import os
from recording import Recording

def one_hot_encode(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]

class DataHandler(object):

    def __init__(self):
        self.speaker_info = {'03': ('m', 31), '08': ('f', 34), '09': ('f', 21), '10': ('m', 32), '11': ('m', 26),
                             '12': ('m', 30), '13': ('f', 32), '14': ('f', 35), '15': ('m', 25), '16': ('f', 31)}
        self.emotion_map = {'A': 'fear', 'E': 'disgust', 'W': 'anger', 'L': 'boredom',
                            'F': 'happiness', 'N': 'neutral', 'T': 'sadness'}

    def emotion_from_ID(self,ID,num=True):
        if num:
            return list(self.emotion_map.keys()).index(ID[5])
        else:
            return self.emotion_map[ID[5]]

    def build_feature_data(self, condition, wav_path = './EmoDB/wav/', laut_path = './EmoDB/lablaut/'):
        assert condition in [1,2,3], 'unknown condition {}'.format(condition)

        data_available = sorted([s.split('.')[0] for s in os.listdir(wav_path) if s.endswith('.wav')])

        df = pd.DataFrame()
        row_list = []
        for ID in tqdm(data_available):
            rec = Recording(ID)
            if len(rec.df_tags) <= 1:
                print('Excluded {} (empty tag-file)'.format(ID))
                continue

            if condition == 1:
                row_list.append({
                    'ID': ID,
                    'feature_vec': rec.get_features(per_phoneme=False)
                })
            if condition in [2,3]:
                for i,row in rec.df_tags.iterrows():
                    row_list.append({
                        'phoneme': row.phoneme,
                        'ID': ID,
                        'feature_vec': rec.get_features(row.t_start,row.t_stop)
                    })
                pass

        df = df.append(row_list)
        df['sex'] = df.ID.apply(lambda x: self.speaker_info[x[:2]][0])
        df['speaker_id'] = df.ID.apply(lambda x: x[:2])
        df['age'] = df.ID.apply(lambda x: self.speaker_info[x[:2]][1])
        df['emotion_name'] = df.ID.apply(lambda x: self.emotion_from_ID(x,num=False))
        df['emotion_label'] = df.ID.apply(lambda x: self.emotion_from_ID(x))
        return df
