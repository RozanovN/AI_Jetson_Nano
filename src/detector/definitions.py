from enum import Enum


class GomokuPiece(Enum):
    WP = "white piece"
    BP = "black piece"
    NP = "no piece"

BOARD_LENGTH = BOARD_WIDTH = 19
BOARD_SIZE = BOARD_WIDTH * BOARD_LENGTH
