import numpy as np
import chess
import chess.svg
import pdb

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
encoded_pos = np.load('../model/encoded_pos.npy')
X_test = np.load('../model/xtest.npy', allow_pickle=True)
X_recons = np.load('../model/recon_pos.npy')
recon_dif = np.load('../model/difference250251recons.npy')
fen_recons = tensor_to_fen(X_recons[:300])
fen_recon_dif = tensor_to_fen(recon_dif)
pdb.set_trace()
print("embedding0: ", list(X_recons[0]))
print("embedding251: ", list(X_recons[251]))
print("orig0: ", X_test[0])
print("orig251: ", X_test[251])
print("rec0: ", fen_recons[0])
print("rec251: ",  fen_recons[251])

def save_svg(filepath, data):
    board = chess.Board(data)
    board_svg = chess.svg.board(board=board)
    with open(filepath, "w") as f:
        f.write(board_svg)

#save_svg("deneme_251.SVG", X_test[251])
#save_svg("deneme_251RECON.SVG", fen_recons[251])
# save_svg("deneme_251_250_difference_RECON.SVG", fen_recon_dif[0])