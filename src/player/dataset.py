import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np

board_size = 15


def parse_xml_to_npy(xml_file):
  tree = ET.parse(xml_file)
  root = tree.getroot() 
  games = root.find('games')
  games_data = []
  
  for game in games.findall('game'):
    move_text = game.find('move').text
    
    if not move_text:
      continue
    
    moves = move_text.split(' ')
    game_id = game.get('id')
    bresult = game.get('bresult')
    rule = game.get('rule')
    
    if rule != '7':
      continue
    
    board_states = []
    board = np.zeros((board_size, board_size), dtype=np.int8)
    
    for turn, move in enumerate(moves):
      row = ord(move[0]) - ord('a')
      col = int(move[1:]) - 1
      board[row][col] = turn % 2 + 1 # 1: black, 2: white
      board_states.append(np.copy(board))
      
    games_data.append({
      'id': game_id,
      'moves': board_states,
      'bresult': bresult,
      'rule': rule,
    })
    
  np.save('gomoku_pro_dataset.npy', games_data)
      
      
    
    