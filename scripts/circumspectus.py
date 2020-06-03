import pdb
import pickle
import pandas as pd 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Model, load_model
from keras.layers import Input, Dense, Flatten, Activation, RepeatVector, Permute, Lambda, merge, BatchNormalization, Masking, concatenate, dot, Dropout
from keras.layers.recurrent import LSTM
from keras import backend as K
from keras.utils import to_categorical
from keras.preprocessing import sequence
from keras.callbacks import Callback
from keract import get_activations
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

master_to_label = {
    'Alekhine': 0,
    'Anand': 1,
    'Botvinnik': 2,
    'Capablanca': 3,
    'Carlsen': 4,
    'Caruana': 5,
    'Fischer': 6,
    'Kasparov': 7,
    'Morphy': 8,
    'Nakamura': 9, 
    'Polgar': 10,
    'Tal': 11
}

def attention_3d_block(hidden_states):
    """
    Many-to-one attention mechanism for Keras.
    @param hidden_states: 3D tensor with shape (batch_size, time_steps, input_dim).
    @return: 2D tensor with shape (batch_size, 128)
    @author: felixhao28.
    """
    hidden_size = int(hidden_states.shape[2])
    # Inside dense layer
    #              hidden_states            dot               W            =>           score_first_part
    # (batch_size, time_steps, hidden_size) dot (hidden_size, hidden_size) => (batch_size, time_steps, hidden_size)
    # W is the trainable weight matrix of attention Luong's multiplicative style score
    score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec')(hidden_states)
    #            score_first_part           dot        last_hidden_state     => attention_weights
    # (batch_size, time_steps, hidden_size) dot   (batch_size, hidden_size)  => (batch_size, time_steps)
    h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state')(hidden_states)
    score = dot([score_first_part, h_t], [2, 1], name='attention_score')
    attention_weights = Activation('softmax', name='attention_weight')(score)
    # (batch_size, time_steps, hidden_size) dot (batch_size, time_steps) => (batch_size, hidden_size)
    context_vector = dot([hidden_states, attention_weights], [1, 1], name='context_vector')
    pre_activation = concatenate([context_vector, h_t], name='attention_output')
    attention_vector = Dense(128, use_bias=False, activation='tanh', name='attention_vector')(pre_activation)
    return attention_vector

# load  and create pos2vec model (non-trainable)
autoencoder = load_model("../model/dense_autoencoder_layer1.h5")

input_position = Input(shape=(448,))

encoder_layer_1 = autoencoder.layers[1]
encoder_layer_2 = autoencoder.layers[2]
encoder_layer_3 = autoencoder.layers[3]
encoder_layer_4 = autoencoder.layers[4]

encoder_layer_1.trainable = False
encoder_layer_2.trainable = False
encoder_layer_3.trainable = False
encoder_layer_4.trainable = False

encoded = encoder_layer_1(input_position)
encoded = encoder_layer_2(encoded)
encoded = encoder_layer_3(encoded)
encoded = encoder_layer_4(encoded)

pos2vec = Model(input_position, encoded)

# # encode and preprocess games
# def encode_and_preprocess(data):
#     encoded_data = [pos2vec.predict(game) for game in data]
#     # pdb.set_trace()
#     processed_data = []
#     max_game = -1
#     for game in encoded_data:
#         # game.shape[0] mod 2 == 0 -> white won
#         # game.shape[0] mod 2 == 1 -> black won (do not use last move)
#         temp = []
#         if max_game < game.shape[0]:
#             max_game = game.shape[0]
#         for i in range(game.shape[0]//2): 
#             temp.append(game[2*i+1,:])
            
#         processed_data.append(np.array(temp))
#     return processed_data

# encode and preprocess games (tuple of consecutive moves)
def encode_and_preprocess_batch(batch):
    encoded_batch = [pos2vec.predict(game) for game in batch]
    processed_batch = []
    for game in encoded_batch:
        # game.shape[0] mod 2 == 0 -> white won
        # game.shape[0] mod 2 == 1 -> black won (do not use last move)
        temp = []
        for i in range(game.shape[0]//2): 
            temp.append(game[2*i:2*(i+1),:])
        processed_batch.append(np.array(temp))
    return processed_batch


# load dataset
with open("../data/white_dataset_2.pkl", "rb") as f:
    dataset = pickle.load(f)

# preprocess dataset
labels = []
games = []
raw_games = []
invalid_data_count = 0
for idx, data in enumerate(dataset):
    game = np.array(data[1])
    temp = game.reshape((-1, np.prod(game.shape[1:])))
    label = master_to_label[data[0]]
    if temp.shape[0] == 1: # only empty board is present in the game
        invalid_data_count += 1
        # print("Error at idx: ", idx)
    else: 
        raw_games.append(game)
        games.append(temp)
        labels.append(label)
print("ERROR occured for ", invalid_data_count, " games.")


# split train(80%), validation(10%) and test(10%) datasets in a stratified manner
train_vald_data_games, test_data_games, train_vald_raw_games, test_raw_games, train_vald_labels_, test_labels = \
    train_test_split(games, raw_games, labels, shuffle=True, random_state=35, stratify=labels, test_size=0.1)

np.save("../attention/test_labels.npy", test_labels)
np.save("../attention/test_raw_games.npy", test_raw_games)

# encode data
train_vald_data = encode_and_preprocess(train_vald_data_games)
test_data = encode_and_preprocess(test_data_games)

# pad sequences to maximum length
train_vald_data = sequence.pad_sequences(train_vald_data, value=0, dtype='float32')
test_data = sequence.pad_sequences(test_data, value=0, maxlen=182, dtype='float32')

# one hot encode labels
train_vald_labels = to_categorical(train_vald_labels_, num_classes=12)
test_labels = to_categorical(test_labels, num_classes=12)

class VisualiseAttentionMap(Callback):

    def on_epoch_end(self, epoch, logs=None):
        attention_map = get_activations(model, test_data, layer_name='attention_weight')['attention_weight']

        plt.plot(attention_map[11])

        iteration_no = str(epoch+1).zfill(3)
        plt.title(f'Iteration {iteration_no} / {30}')
        plt.xlabel("Board Positions")
        plt.ylabel("Signal Intensity")
        plt.savefig(f"../model/final_epoch_{iteration_no}.png")
        plt.close()
        plt.clf()
def plot_attention(idx):
    plt.plot(attention_map[idx])
    plt.title(f'Test data: {idx}')
    plt.xlabel("Board Positions")
    plt.ylabel("Signal Intensity")
    plt.savefig(f"../model/test_{idx}.png")
    plt.close()


# # define model
# max_length = np.max([x.shape[0] for x in train_vald_data])
# print("Max Length sequence in train and validation data: ", max_length)
# input1 = Input(shape=(max_length,42))
# masked_input1 = Masking(mask_value = 0)(input1)
# features2 = LSTM(32, return_sequences=True)(masked_input1)
# features3 = BatchNormalization()(features2)
# features4 = attention_3d_block(features3)
# features4 = Dense(32, activation='relu')(features4)
# # features4 = Dropout(0.2)(features4)
# output = Dense(12,activation='softmax')(features4)

# model = Model(inputs=input1, outputs = output)
# print(model.summary())

# define model
max_length = np.max([x.shape[0] for x in train_vald_data])
print("Max Length sequence in train and validation data: ", max_length)
input1 = Input(shape=(max_length,42))
input2 = Input(shape=(max_length,42))
masked_input1 = Masking(mask_value = 0)(input1)
masked_input2 = Masking(mask_value = 0)(input2)
catted_features = Concatenate(axis=1)([masked_input1, masked_input2])
features1 = Dense(42)(catted_features)
features2 = LSTM(32, return_sequences=True)(features1)
features3 = BatchNormalization()(features2)
features4 = attention_3d_block(features3)
features4 = Dense(32, activation='relu')(features4)
# features4 = Dropout(0.2)(features4)
output = Dense(12,activation='softmax')(features4)
model = Model(inputs=[input1, input2], outputs = output)
print(model.summary())

weights = class_weight.compute_class_weight('balanced', np.unique(train_vald_labels_), train_vald_labels_)
model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
model.fit(train_vald_data, train_vald_labels, validation_split = 0.2, epochs = 30, batch_size=128, class_weight=weights, callbacks=[VisualiseAttentionMap()])
model.save("../model/final_2.h5")
attention_map = get_activations(model, test_data, layer_name='attention_weight')['attention_weight']
np.save("../attention/attention_map.npy", attention_map)
pdb.set_trace()



