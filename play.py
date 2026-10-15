import argparse
from pathlib import Path
from typing import Optional
import chess
import torch

from src.config import get_device, MODELS_DIR
from src.network import ChessValueNetwork
from src.agent import select_best_move


def parse_player_move(board: chess.Board, move_text: str) -> Optional[chess.Move]:
    """
    Parses user input string into a validated legal chess.Move.
    Accepts standard UCI format (e.g. 'e2e4', 'e7e8q') or SAN format (e.g. 'e4', 'Nf3', 'O-O').

    Args:
        board: Current game board context for validation.
        move_text: Raw string input from user.

    Returns:
        Optional[chess.Move]: The legal move object, or None if invalid/illegal.
    """
    clean_text = move_text.strip()
    if not clean_text:
        return None

    # Try UCI notation first
    try:
        candidate_move = chess.Move.from_uci(clean_text)
        if candidate_move in board.legal_moves:
            return candidate_move
    except ValueError:
        pass

    # Try Standard Algebraic Notation (SAN)
    try:
        candidate_move = board.parse_san(clean_text)
        if candidate_move in board.legal_moves:
            return candidate_move
    except ValueError:
        pass

    return None


def render_board(board: chess.Board, player_is_black: bool = False) -> str:
    """
    Renders terminal-friendly ASCII representation of the chess board.
    Flipped if the human plays as Black.
    """
    board_str = str(board if not player_is_black else board.transform(chess.flip_vertical))
    return board_str


def play_interactive_game(
    human_color: chess.Color,
    weights_path: Optional[Path] = None,
) -> None:
    """
    Executes an interactive game loop between human user and AI model.

    Args:
        human_color: chess.WHITE or chess.BLACK.
        weights_path: Optional path to saved model checkpoint (.pt).
    """
    device = get_device()
    model = ChessValueNetwork().to(device)

    if weights_path is not None and weights_path.is_file():
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded model weights from {weights_path}")
    else:
        print("Using model with initial random weights (untrained).")

    model.eval()
    board = chess.Board()

    print("\n--- Chess AI Match Started ---")
    print(f"Human: {'White' if human_color == chess.WHITE else 'Black'}")
    print(f"AI:    {'Black' if human_color == chess.WHITE else 'White'}")
    print("Input formats: UCI (e.g., 'e2e4') or SAN (e.g., 'e4', 'Nf3'). Type 'quit' to exit.\n")

    while not board.is_game_over():
        print("-" * 30)
        print(f"Turn: {'White' if board.turn == chess.WHITE else 'Black'} (Move {board.fullmove_number})")
        print(render_board(board, player_is_black=(human_color == chess.BLACK)))
        print("-" * 30)

        if board.turn == human_color:
            # Human turn
            while True:
                user_input = input("Enter your move: ").strip()
                if user_input.lower() in ("quit", "exit"):
                    print("Match terminated by user.")
                    return

                move = parse_player_move(board, user_input)
                if move is not None:
                    board.push(move)
                    break
                else:
                    print(f"Invalid or illegal move: '{user_input}'. Legal moves: {[board.san(m) for m in board.legal_moves]}")
        else:
            # AI turn
            print("AI is evaluating candidate positions...")
            ai_move = select_best_move(board, model, device)
            if ai_move is None:
                break
            san_repr = board.san(ai_move)
            board.push(ai_move)
            print(f"AI played: {san_repr} ({ai_move.uci()})")

    # Final game state
    print("\n" + "=" * 30)
    print("GAME OVER")
    print(render_board(board, player_is_black=(human_color == chess.BLACK)))
    print("=" * 30)

    if board.is_checkmate():
        winner = "Black" if board.turn == chess.WHITE else "White"
        print(f"Result: Checkmate. {winner} wins ({board.result()}).")
    elif board.is_stalemate():
        print(f"Result: Stalemate. Draw ({board.result()}).")
    elif board.is_insufficient_material():
        print(f"Result: Draw by insufficient material ({board.result()}).")
    else:
        print(f"Result: Game drawn ({board.result()}).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play chess against the AI engine.")
    parser.add_argument(
        "--color",
        type=str,
        choices=["white", "black"],
        default="white",
        help="Player color choice (default: white)",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Path to trained PyTorch weights file (.pt)",
    )

    args = parser.parse_args()
    selected_color = chess.WHITE if args.color.lower() == "white" else chess.BLACK

    play_interactive_game(selected_color, weights_path=args.weights)
