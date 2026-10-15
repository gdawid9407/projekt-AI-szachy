from typing import Optional
import chess
import torch
import torch.nn as nn

from src.encoding import encode_board


def evaluate_moves(
    board: chess.Board,
    model: nn.Module,
    device: torch.device,
) -> tuple[list[chess.Move], torch.Tensor]:
    """
    Evaluates all legal moves from the current board position using batch inference.

    Args:
        board: The current chess.Board state.
        model: Evaluator neural network producing (B, 1) scalar evaluations.
        device: Target compute device (cuda / mps / cpu).

    Returns:
        tuple[list[chess.Move], torch.Tensor]:
            - List of legal candidate moves.
            - 1D tensor of shape (N,) containing evaluations for each candidate move.
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return [], torch.empty(0, device=device)

    # 1. Generate child states for all legal moves
    tensors: list[torch.Tensor] = []
    for move in legal_moves:
        board.push(move)
        tensors.append(encode_board(board))
        board.pop()

    # Batch tensor construction: (N_legal, 14, 8, 8)
    batch = torch.stack(tensors).to(device)

    # Batch inference: evaluations strictly computed without tracking gradients
    model.eval()
    with torch.no_grad():
        evaluations = model(batch).squeeze(-1)  # Tensor shape: (N_legal,)

    return legal_moves, evaluations


def select_best_move(
    board: chess.Board,
    model: nn.Module,
    device: torch.device,
) -> Optional[chess.Move]:
    """
    Selects the optimal legal move for the active player based on 1-ply value evaluations.

    Args:
        board: Current game board.
        model: Value network.
        device: Compute accelerator.

    Returns:
        Optional[chess.Move]: Selected move, or None if no legal moves exist.
    """
    legal_moves, evaluations = evaluate_moves(board, model, device)
    if not legal_moves:
        return None

    # TODO: Implement move selection logic based on board.turn:
    # 1. If White is moving (board.turn == chess.WHITE):
    #    Select the move that maximizes the evaluation score.
    # 2. If Black is moving (board.turn == chess.BLACK):
    #    Select the move that minimizes the evaluation score.
    # 3. Return the chosen chess.Move instance.
    raise NotImplementedError("Complete select_best_move implementation.")


if __name__ == "__main__":
    from src.config import get_device
    from src.network import ChessValueNetwork

    device = get_device()
    network = ChessValueNetwork().to(device)

    test_board = chess.Board()
    try:
        best_move = select_best_move(test_board, network, device)
        print(f"Selected move from initial position: {best_move}")
    except NotImplementedError as e:
        print(f"Agent scaffolding ready. Pending completion: {e}")
