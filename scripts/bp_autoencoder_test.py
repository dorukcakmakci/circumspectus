import pdb
import numpy as np
import pandas as pd
from keras.models import Model, load_model
from keras.layers import Input


def tensor_to_fen(inputtensor):
    pieces_str = "PNBRQK"
    pieces_str += pieces_str.lower()
    
    maxnum = len(inputtensor)
    
    outputbatch = []
    for i in range(maxnum):
        fenstr = ""
        for rownr in range(8):
            spaces = 0
            for colnr in range(8):
                for lay in range(6):                    
                    if inputtensor[i,rownr,colnr,lay] == 1:
                        if spaces > 0:
                            fenstr += str(spaces)
                            spaces = 0
                        fenstr += pieces_str[lay]
                        break
                    elif inputtensor[i,rownr,colnr,lay] == -1:
                        if spaces > 0:
                            fenstr += str(spaces)
                            spaces = 0
                        fenstr += pieces_str[lay+6]
                        break
                    if lay == 5:
                        spaces += 1
            if spaces > 0:
                fenstr += str(spaces)
            if rownr < 7:
                fenstr += "/"
        if inputtensor[i,0,0,6] == 1:
            fenstr += " w"
        else:
            fenstr += " b"
        outputbatch.append(fenstr)
    
    return outputbatch
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

autoencoder = load_model("../model/autoencoder_weights.h5")

#create the encoder

input_position = Input(shape=(448,))

encoder_layer_1 = autoencoder.layers[1]
encoder_layer_2 = autoencoder.layers[2]
encoder_layer_3 = autoencoder.layers[3]
encoder_layer_4 = autoencoder.layers[4]

encoded = encoder_layer_1(input_position)
encoded = encoder_layer_2(encoded)
encoded = encoder_layer_3(encoded)
encoded = encoder_layer_4(encoded)

encoder = Model(input_position, encoded)

encoder.summary()

#create the decoder

encoding_dim = 42  
encoded_input = Input(shape=(encoding_dim,))


decoder_layer_1 = autoencoder.layers[-4]
decoder_layer_2 = autoencoder.layers[-3]
decoder_layer_3 = autoencoder.layers[-2]
decoder_layer_4 = autoencoder.layers[-1]

decoded = decoder_layer_1(encoded_input)
decoded = decoder_layer_2(decoded)
decoded = decoder_layer_3(decoded)
decoded = decoder_layer_4(decoded)

decoder = Model(encoded_input, decoded)
decoder.summary()

testinput = batchtotensor(X_test)
testinput_flat = testinput.reshape((-1, np.prod(testinput.shape[1:])))
encoded_pos = encoder.predict(testinput_flat)
decoded_pos = decoder.predict(encoded_pos)
decoded_pos = decoded_pos.reshape((-1,8,8,7))
recon_pos = np.round_(decoded_pos)

fen_recon_pos = tensor_to_fen(recon_pos)
fen_orig_pos = X_test

pdb.set_trace()


##Draw selected reconstructions.
import chess
import chess.svg

board = chess.Board("8/8/8/8/4N3/8/8/8 w - - 0 1")
squares = board.attacks(chess.E4)
chess.svg.board(board=board, squares=squares)  