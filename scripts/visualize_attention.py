import pdb
import matplotlib.pyplot as plt
import numpy as np
import chess
import chess.svg

label_to_master = {
    "0": 'Alekhine',
    "1": 'Anand',
    "2": 'Botvinnik',
    "3": 'Capablanca',
    "4": 'Carlsen',
    "5": 'Caruana',
    "6": 'Fischer',
    "7": 'Kasparov',
    "8": 'Morphy',
    "9": 'Nakamura', 
    "10": 'Polgar',
    "11": 'Tal'
}

def tensor_to_fen(inputtensor):
    pieces_str = "PNBRQK"
    pieces_str += pieces_str.lower()
    
    fenstr = ""
    for rownr in range(8):
        spaces = 0
        for colnr in range(8):
            for lay in range(6):                    
                if inputtensor[rownr,colnr,lay] == 1:
                    if spaces > 0:
                        fenstr += str(spaces)
                        spaces = 0
                    fenstr += pieces_str[lay]
                    break
                elif inputtensor[rownr,colnr,lay] == -1:
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
    if inputtensor[0,0,6] == 1:
        fenstr += " w"
    else:
        fenstr += " b"
    return  fenstr

# load information 
test_labels = np.load("../attention/test_labels.npy")
test_games = np.load("../attention/test_raw_games.npy", allow_pickle=True)
test_pred_prob = np.load("../attention/test_pred_prob.npy")
attention_map = np.load("../attention/attention_map.npy")

# find correctly classified instances
test_preds = np.argmax(test_pred_prob, axis=1)
correct_idx = test_preds == test_labels
correct_idx = np.where(correct_idx == True)[0] 
correct_labels = test_preds[correct_idx]
correct_games = test_games[correct_idx]
correct_attn_map = attention_map[correct_idx]

def visualize_attn_map(idx):
    game_len = correct_games[idx].shape[0] // 2
    attn_map = correct_attn_map[idx]
    label = correct_labels[idx]
    plt.plot(attn_map)
    plt.title(f"Game Length: {game_len} , Player: {label_to_master[str(label)]}")
    plt.savefig(f"../attention/new/game_{idx}_player_{label_to_master[str(label)]}.png")
    temp = attn_map.argsort()[-20:][::-1]
    return  2 * (temp - (182 - game_len) + 1) - 1

def visualize_game(filename, game_idx, move_idx):
    game = correct_games[game_idx]
    move = game[move_idx]
    fen = tensor_to_fen(move)
    board = chess.Board(fen)
    board_svg = chess.svg.board(board=board)
    with open("../attention/new/" + filename, "w") as f:
        f.write(board_svg)

max_pts = []
for i in range(len(correct_idx)):
    max_pts.append(visualize_attn_map(i))
    plt.close()


pdb.set_trace()