import numpy as np
from keras.layers import Input,Dense,Dropout
from keras.models import Model
import keras

class MultiLayerPerceptron(object):

    def __init__(self,layer_sizes):
        self.layer_sizes = layer_sizes
        self._build_model()

    def _build_model(self):
        inputs = Input(shape=(self.layer_sizes[0],))
        state = Dropout(rate=0.5)(inputs)
        for l in self.layer_sizes[1:-1]:
            state = Dense(l, activation='tanh')(state)
            state = Dropout(rate=0.2)(state)
        predictions = Dense(self.layer_sizes[-1], activation='softmax')(state)
        self.model = Model(inputs=inputs, outputs=predictions)
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def fit(self,X,y,val_data,**kwargs):
        self.model.fit(X,y,batch_size=32,verbose=0,callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',patience=20,restore_best_weights=True)],validation_data=val_data,**kwargs)

    def predict_proba(self,X,**kwargs):
        return self.model.predict(X,**kwargs)

    def predict(self,X,**kwargs):
        return np.argmax(self.model.predict(X,**kwargs),axis=1)
