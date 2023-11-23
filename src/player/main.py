import numpy as np

from dataset import parse_xml_to_npy

def main():
  # parse_xml_to_npy('datasets/renju_net/renjunet_v10_20231122.xml')
  data = np.load('datasets/renju_net/gomoku_pro_dataset.npy', allow_pickle=True)

if __name__ == '__main__':
  main()