import pandas as pd
import numpy as np
import glob, os, re
from keras.utils import Sequence, to_categorical
from keras.preprocessing.sequence import pad_sequences

board_size = 20
directory = 'datasets/gomocup/freestyle2_2022'

class GomokuDataset(Sequence):
  def __init__(self, games, batch_size=32, board_size=20, **kwargs):
    self.games = games
    self.batch_size = batch_size
    self.board_size = board_size
    
  def __len__(self):
    return np.ceil(len(self.games) / self.batch_size).astype(int)
  
  def __getitem__(self, idx):
    batch = self.games[idx * self.batch_size:(idx + 1) * self.batch_size]
    
    max_len = max([len(game['board_states']) for game in batch])
    x = pad_sequences([game['board_states'] for game in batch], maxlen=max_len, padding='post', value=0, dtype=np.int8)
    y = pad_sequences([to_categorical(label, num_classes=self.board_size**2) for label in [game['labels'] for game in batch]], maxlen=max_len, padding='post', value=0)
    # y = pad_sequences([game['labels'] for game in batch], maxlen=max_len, padding='post', value=0)
    x = np.expand_dims(x, axis=-1)

    return x, y

def parse_gomocup_dataset():
  games_data = []
  
  for filepath in glob.glob(os.path.join(directory, '*.psq')):
    with open(filepath, 'r') as file:
      filename = os.path.basename(filepath).split('.')[0]
      id, result = filename.rsplit('_', 1)
      board_states = []
      labels = []
      board = np.zeros((board_size, board_size), dtype=np.int8)
      
      for idx, line in enumerate(file.readlines()[1:]):
        if re.match(r'^\d+,\d+,\d+$', line.strip()):
          col_str, row_str = line.strip().split(',')[:2]
          col, row = int(col_str) - 1, int(row_str) - 1
          board[col][row] = 1 - (idx % 2) * 2 # 1: black, -1: white
          board_states.append(np.copy(board))
          labels.append(col * board_size + row)
        else:
          break
      
      games_data.append({
        'id': id,
        'board_states': board_states,
        'labels': labels,
        'result': result,
      })  
  
  np.save('datasets/gomocup_2022_dataset.npy', games_data) 


def main():
  parse_gomocup_dataset()

# if __name__ == '__main__':
  # main()

    