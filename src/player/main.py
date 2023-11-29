import numpy as np
from sklearn.model_selection import train_test_split

from model import GomokuPlayer
from dataset import GomokuDataset

data = np.load('datasets/gomocup_2022_dataset.npy', allow_pickle=True)
batch_size = 32
model_config = {
  "board_size": 20,
  "num_epoch": 10,
  "batch_size": batch_size,
}
model = GomokuPlayer(**model_config)

train_data, val_test_data = train_test_split(data, test_size=0.2, random_state=266)
val_data, test_data = train_test_split(val_test_data, test_size=0.5, random_state=266)

train_dataset, val_dataset, test_dataset = GomokuDataset(train_data, **model_config), GomokuDataset(val_data, batch_size), GomokuDataset(test_data, batch_size)

def train():
  model.compile()
  model.fit(train_dataset)
  model.evaluate(val_dataset)

def main():
  train()
  
  
if __name__ == '__main__':
  main()