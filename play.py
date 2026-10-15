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


UNICODE_PIECES = {
    chess.PAWN: ("♙", "♟"),
    chess.KNIGHT: ("♘", "♞"),
    chess.BISHOP: ("♗", "♝"),
    chess.ROOK: ("♖", "♜"),
    chess.QUEEN: ("♕", "♛"),
    chess.KING: ("♔", "♚"),
}


def render_board(board: chess.Board, player_is_black: bool = False) -> str:
    """
    Renders an enhanced Unicode box-drawing chessboard with ranks, files,
    piece glyphs, and alternating dark/light square patterns.
    """
    ranks = range(8) if player_is_black else range(7, -1, -1)
    files = range(7, -1, -1) if player_is_black else range(8)

    file_letters = ["a", "b", "c", "d", "e", "f", "g", "h"]
    file_labels = "    " + "   ".join(file_letters[f] for f in files)
    top_border = "  ┌" + "───┬" * 7 + "───┐"
    mid_border = "  ├" + "───┼" * 7 + "───┤"
    bot_border = "  └" + "───┴" * 7 + "───┘"

    last_rank_index = 0 if not player_is_black else 7
    lines = [file_labels, top_border]

    for rank in ranks:
        row_str = f"{rank + 1} │"
        for file in files:
            sq = chess.square(file, rank)
            piece = board.piece_at(sq)
            if piece is None:
                symbol = "·" if (rank + file) % 2 == 1 else " "
            else:
                symbols = UNICODE_PIECES[piece.piece_type]
                symbol = symbols[0] if piece.color == chess.WHITE else symbols[1]
            row_str += f" {symbol} │"

        row_str += f" {rank + 1}"
        lines.append(row_str)
        if rank != last_rank_index:
            lines.append(mid_border)

    lines.append(bot_border)
    lines.append(file_labels)
    return "\n".join(lines)


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

    white_symbols = "♔ ♕ ♖ ♗ ♘ ♙"
    black_symbols = "♚ ♛ ♜ ♝ ♞ ♟"

    human_is_white = (human_color == chess.WHITE)
    human_pieces = white_symbols if human_is_white else black_symbols
    ai_pieces = black_symbols if human_is_white else white_symbols
    human_side = "White" if human_is_white else "Black"
    ai_side = "Black" if human_is_white else "White"

    print("\n" + "=" * 48)
    print("             CHESS AI MATCH STARTED             ")
    print("=" * 48)
    print(f" Player : {human_side:<5} [ {human_pieces} ] (Bottom)")
    print(f" AI     : {ai_side:<5} [ {ai_pieces} ] (Top)")
    print("-" * 48)
    print(" Move notation: UCI ('e2e4') or SAN ('e4', 'Nf3')")
    print(" Commands     : type 'quit' or 'exit' to quit")
    print("=" * 48 + "\n")

    while not board.is_game_over():
        current_actor = "Player" if board.turn == human_color else "AI"
        turn_color = "White" if board.turn == chess.WHITE else "Black"
        active_symbols = white_symbols if board.turn == chess.WHITE else black_symbols

        print("-" * 40)
        print(f" Move {board.fullmove_number} | Turn: {turn_color} ({current_actor}) [ {active_symbols} ]")
        print(render_board(board, player_is_black=(human_color == chess.BLACK)))
        print("-" * 40)

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
