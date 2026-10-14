import numpy as np
import torch
import chess
from src.config import BOARD_CHANNELS, BOARD_SIZE

# Channel mapping indices
# 0-5: White pieces (P, N, B, R, Q, K)
# 6-11: Black pieces (P, N, B, R, Q, K)
PIECE_TO_CHANNEL = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}


def encode_board(board: chess.Board) -> torch.Tensor:
    """
    Converts a chess.Board state into a (14, 8, 8) float32 PyTorch tensor.

    Channels:
      0-5:   White pieces (P, N, B, R, Q, K)
      6-11:  Black pieces (p, n, b, r, q, k)
      12:    Active player plane (all 1.0 if White to move, all 0.0 if Black)
      13:    Castling rights plane
    """
    # Initialize zero-filled array of shape (14, 8, 8)
    tensor = np.zeros((BOARD_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    # 1. Populate piece planes (Channels 0 - 11)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            # Rank (0-7) maps to row, file (0-7) maps to col
            rank = chess.square_rank(square)
            file = chess.square_file(square)

            channel_offset = 0 if piece.color == chess.WHITE else 6
            channel = channel_offset + PIECE_TO_CHANNEL[piece.piece_type]
            tensor[channel, rank, file] = 1.0

    # 2. Active player plane (Channel 12)
    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1.0

    # 3. Castling rights plane (Channel 13)
    # File 0 is a-file (Queenside), File 7 is h-file (Kingside)
    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[13, 0, 7] = 1.0  # h1: White Kingside
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[13, 0, 0] = 1.0  # a1: White Queenside
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[13, 7, 7] = 1.0  # h8: Black Kingside
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[13, 7, 0] = 1.0  # a8: Black Queenside

    return torch.from_numpy(tensor)
