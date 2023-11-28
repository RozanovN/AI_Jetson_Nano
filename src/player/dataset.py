import pandas as pd
import numpy as np
import glob, os, re


board_size = 20
directory = 'datasets/gomocup/freestyle2_2022'


def parse_gomocup_dataset():
  games_data = []
  
  for filepath in glob.glob(os.path.join(directory, '*.psq')):
    with open(filepath, 'r') as file:
      filename = os.path.basename(filepath).split('.')[0]
      id, result = filename.rsplit('_', 1)
      board_states = []
      board = np.zeros((board_size, board_size), dtype=np.int8)
      
      for idx, line in enumerate(file.readlines()[1:]):
        if re.match(r'^\d+,\d+,\d+$', line.strip()):
          col, row = line.strip().split(',')[:2]
          board[int(row)-1][int(col)-1] = idx % 2 + 1 # 1: black, 2: white
          board_states.append(np.copy(board))
        else:
          break
      
      games_data.append({
        'id': id,
        'moves': board_states,
        'result': result,
      })  
  
  np.save('datasets/gomocup_2022_dataset.npy', games_data) 


def main():
  parse_gomocup_dataset()

if __name__ == '__main__':
  main()

    