import os
import pdb 
import sys
import pickle
import chess.pgn

import pandas as pd
import numpy as np

sys.path.insert(1,"../lib/Chess2Vec/")

def fentotensor(fen):
    #take fens as batches and convert them to 8x8x7 tensors
    pieces_str = "PNBRQK"
    pieces_str += pieces_str.lower()
    pieces = set(pieces_str)
    valid_spaces = set(range(1,9))
    pieces_dict = {pieces_str[0]:1, pieces_str[1]:2, pieces_str[2]:3, pieces_str[3]:4,
                    pieces_str[4]:5, pieces_str[5]:6,
                    pieces_str[6]:-1, pieces_str[7]:-2, pieces_str[8]:-3, pieces_str[9]:-4, 
                    pieces_str[10]:-5, pieces_str[11]:-6}

    boardtensor = np.zeros((8,8,7))
    
    inputliste = fen.split()
    rownr = 0
    colnr = 0
    for i, c in enumerate(inputliste[0]):
        if c in pieces:
            boardtensor[rownr, colnr, np.abs(pieces_dict[c])-1] = np.sign(pieces_dict[c])
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
                boardtensor[i, j, 6] = 1
    else:
        for i in range(8):
            for j in range(8):
                boardtensor[i, j, 6] = -1
  
    return boardtensor

pgn_file_root = "../data/Raw_game/Raw_game/"

dataset_file = pd.read_csv("../data/game_data.csv")
print(dataset_file.head())
print("Grandmaster Color Counts\nBlack: ", len(dataset_file[dataset_file.color=="Black"].index), "\nWhite: ", len(dataset_file[dataset_file.color=="White"].index))
dataset = dataset_file[dataset_file.color=="White"]
grandmaster_names = np.unique(dataset.player)
print(grandmaster_names)
for gm in grandmaster_names:
    print(len(dataset[dataset.player==gm]))
pdb.set_trace()

data = []
for row in dataset.iterrows():
    master = row[1].player
    game_name = row[1].file_name
    game_path = os.path.join(pgn_file_root, master,game_name)
    pgn = open(game_path)
    game = chess.pgn.read_game(pgn)
    board = game.board()
    flag = True
    boards = []
    boards.append(fentotensor(board.fen())) # first append empty board to each game
    for move in game.mainline_moves():
        board.push(move)
        # convert FEN to tensor here
        boards.append(fentotensor(board.fen()))
    data.append((master, boards))
pdb.set_trace()

