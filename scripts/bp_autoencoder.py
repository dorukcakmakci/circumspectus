import numpy as np
import pandas as pd
import pdb
import random
from keras.layers import Dense, Input
from keras.models import Model

def batchtotensor(inputbatch):
    #take fens as batches and convert them to 8x8x7 tensors
    pieces_str = "PNBRQK"
    pieces_str += pieces_str.lower()
    pieces = set(pieces_str)
    valid_spaces = set(range(1,9))
    pieces_dict = {pieces_str[0]:1, pieces_str[1]:2, pieces_str[2]:3, pieces_str[3]:4,
                    pieces_str[4]:5, pieces_str[5]:6,
                    pieces_str[6]:-1, pieces_str[7]:-2, pieces_str[8]:-3, pieces_str[9]:-4, 
                    pieces_str[10]:-5, pieces_str[11]:-6}

    maxnum = len(inputbatch)
    boardtensor = np.zeros((maxnum, 8, 8,7))
    
    for num, inputstr in enumerate(inputbatch):
        inputliste = inputstr.split()
        rownr = 0
        colnr = 0
        for i, c in enumerate(inputliste[0]):
            if c in pieces:
                boardtensor[num, rownr, colnr, np.abs(pieces_dict[c])-1] = np.sign(pieces_dict[c])
                colnr = colnr + 1
            elif c == '/':  # new row
                rownr = rownr + 1
                colnr = 0
            elif int(c) in valid_spaces:
                colnr = colnr + int(c)
            else:
                raise ValueError("invalid fenstr at index: {} char: {}".format(i, c))
        
        if inputliste[1] == "w":
            for i in range(8):
                for j in range(8):
                    boardtensor[num, i, j, 6] = 1
        else:
            for i in range(8):
                for j in range(8):
                    boardtensor[num, i, j, 6] = -1
  
    return boardtensor

df = pd.read_csv("../data/autoencoder_data/fics_fen_2_2M_rnn.csv", header=None, sep=";", names=["FEN"])
X_data = df["FEN"]

X_train = X_data[:2048000]
X_test =  X_data[2048000:]

idx = np.arange(X_train.shape[0])
np.random.shuffle(idx)
X_train = X_train[idx]

#train first layer of autoencoder:

def myGenerator():
    while 1:
        for i in range(16000): # 16000 * 128 = 2048000 -> # of training samples
            ret = batchtotensor(X_train[i*128:(i+1)*128])
            ret = ret.reshape((128, np.prod(ret.shape[1:])))
            yield (ret, ret)

my_generator = myGenerator()
testtensor = batchtotensor(X_test)
validdata = testtensor.reshape((len(X_test), np.prod(testtensor.shape[1:])))



encoding_dim = 42 



input_pos = Input(shape=(448,))



encoded = Dense(300, activation='tanh')(input_pos)
encoded = Dense(200, activation='tanh')(encoded)
encoded = Dense(150, activation='tanh')(encoded)
encoded = Dense(encoding_dim, activation='tanh')(encoded)



decoded = Dense(150, activation='tanh')(encoded)
decoded = Dense(200, activation='tanh')(decoded)
decoded = Dense(300, activation='tanh')(decoded)
decoded = Dense(448, activation='tanh')(decoded)



autoencoder1 = Model(input_pos, decoded)
autoencoder1.summary()


autoencoder1.compile(optimizer='adam', loss='mean_squared_error')
print(autoencoder1.summary())
history = autoencoder1.fit_generator(my_generator, steps_per_epoch = 16000, epochs = 50, verbose=1, 
              validation_data=(validdata,validdata), workers=1)
autoencoder1.save('../model/dense_autoencoder_layer1.h5')


pdb.set_trace()