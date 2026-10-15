import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple, Union
import chess.pgn
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.config import BOARD_CHANNELS, BOARD_SIZE, DATA_DIR
from src.encoding import encode_board

POSITIONS_FILENAME = "positions.dat"
TARGETS_FILENAME = "targets.dat"
METADATA_FILENAME = "metadata.json"


def parse_game_result(result_str: str) -> Optional[float]:
    """
    Maps PGN result string to an absolute evaluation target:
      '1-0'     -> +1.0 (White win)
      '0-1'     -> -1.0 (Black win)
      '1/2-1/2' ->  0.0 (Draw)
      Other     -> None (Incomplete / aborted)
    """
    clean = result_str.strip()
    if clean == "1-0":
        return 1.0
    if clean == "0-1":
        return -1.0
    if clean == "1/2-1/2":
        return 0.0
    return None


def is_game_eligible(game: chess.pgn.Game, min_elo: Optional[int] = None) -> bool:
    """
    Validates whether a game satisfies minimum Elo criteria and has a decisive or drawn outcome.
    """
    if parse_game_result(game.headers.get("Result", "*")) is None:
        return False

    if min_elo is not None:
        try:
            w_elo = int(game.headers.get("WhiteElo", "0"))
            b_elo = int(game.headers.get("BlackElo", "0"))
            if w_elo < min_elo or b_elo < min_elo:
                return False
        except ValueError:
            return False

    return True


def count_streamable_positions(
    pgn_paths: List[Path],
    max_games: Optional[int] = None,
    skip_initial_plies: int = 4,
    min_elo: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Performs a fast first pass over PGN files to calculate exact buffer dimensions for memory mapping.
    """
    valid_games = 0
    total_positions = 0

    for path in pgn_paths:
        if not path.is_file():
            continue
        with open(path, "r", encoding="utf-8", errors="replace") as pgn_file:
            while True:
                if max_games is not None and valid_games >= max_games:
                    break

                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break

                if not is_game_eligible(game, min_elo=min_elo):
                    continue

                move_count = sum(1 for _ in game.mainline_moves())
                usable_moves = max(0, move_count - skip_initial_plies)

                if usable_moves > 0:
                    valid_games += 1
                    total_positions += usable_moves

            if max_games is not None and valid_games >= max_games:
                break

    return valid_games, total_positions


def create_memmap_dataset(
    pgn_input: Union[Path, List[Path]],
    output_dir: Path = DATA_DIR,
    max_games: Optional[int] = None,
    skip_initial_plies: int = 4,
    min_elo: Optional[int] = None,
) -> Path:
    """
    Parses one or multiple PGN files and streams extracted board states directly into disk-backed binary files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / METADATA_FILENAME

    paths = [pgn_input] if isinstance(pgn_input, Path) else pgn_input
    paths = [p for p in paths if p.is_file()]

    if not paths:
        raise FileNotFoundError("No valid PGN files found to construct dataset.")

    print(f"Scanning {len(paths)} PGN archive(s) for position count...")
    valid_games, total_positions = count_streamable_positions(
        paths, max_games=max_games, skip_initial_plies=skip_initial_plies, min_elo=min_elo
    )
    print(f"Discovered {valid_games} eligible games with {total_positions:,} positions.")

    if total_positions == 0:
        raise ValueError("Zero eligible positions discovered with specified filtering criteria.")

    pos_path = output_dir / POSITIONS_FILENAME
    tgt_path = output_dir / TARGETS_FILENAME

    # Pre-allocate binary disk files
    pos_memmap = np.memmap(
        pos_path,
        dtype=np.float32,
        mode="w+",
        shape=(total_positions, BOARD_CHANNELS, BOARD_SIZE, BOARD_SIZE),
    )
    tgt_memmap = np.memmap(
        tgt_path,
        dtype=np.float32,
        mode="w+",
        shape=(total_positions, 1),
    )

    current_idx = 0
    progress = tqdm(total=total_positions, desc="Serializing board states", unit="pos")
    games_processed = 0

    for path in paths:
        if max_games is not None and games_processed >= max_games:
            break

        with open(path, "r", encoding="utf-8", errors="replace") as pgn_file:
            while True:
                if max_games is not None and games_processed >= max_games:
                    break

                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break

                if not is_game_eligible(game, min_elo=min_elo):
                    continue

                target = parse_game_result(game.headers.get("Result", "*"))
                if target is None:
                    continue

                board = game.board()
                ply = 0

                for move in game.mainline_moves():
                    board.push(move)
                    ply += 1

                    if ply > skip_initial_plies and current_idx < total_positions:
                        pos_memmap[current_idx] = encode_board(board).numpy()
                        tgt_memmap[current_idx] = target
                        current_idx += 1
                        progress.update(1)

                games_processed += 1

    progress.close()

    # Flush virtual memory pages to persistent disk
    pos_memmap.flush()
    tgt_memmap.flush()

    metadata = {
        "num_samples": total_positions,
        "channels": BOARD_CHANNELS,
        "board_size": BOARD_SIZE,
        "sources": [p.name for p in paths],
        "skip_initial_plies": skip_initial_plies,
        "min_elo": min_elo,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Memory-mapped dataset persisted to {output_dir}. Total samples: {total_positions:,}")
    return meta_path


class ChessMemmapDataset(Dataset):
    """
    PyTorch Dataset reading slices on-demand from disk-backed memory-mapped files.
    Eliminates dataset RAM allocation during training iterations.
    """

    def __init__(self, data_dir: Path = DATA_DIR) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.meta_path = data_dir / METADATA_FILENAME
        self.pos_path = data_dir / POSITIONS_FILENAME
        self.tgt_path = data_dir / TARGETS_FILENAME

        if not self.meta_path.is_file():
            raise FileNotFoundError(
                f"Dataset metadata not found at {self.meta_path}. "
                "Run create_memmap_dataset first."
            )

        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.num_samples = self.metadata["num_samples"]
        self.shape = (
            self.num_samples,
            self.metadata["channels"],
            self.metadata["board_size"],
            self.metadata["board_size"],
        )

        self.positions = np.memmap(
            self.pos_path,
            dtype=np.float32,
            mode="r",
            shape=self.shape,
        )
        self.targets = np.memmap(
            self.tgt_path,
            dtype=np.float32,
            mode="r",
            shape=(self.num_samples, 1),
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.positions[index].copy())
        y = torch.from_numpy(self.targets[index].copy())
        return x, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build memory-mapped binary dataset from PGN files.")
    parser.add_argument(
        "--pgn",
        type=str,
        default=None,
        help="Optional specific PGN file path. If omitted, uses all .pgn files in data/.",
    )
    parser.add_argument("--max-games", type=int, default=500, help="Max games to process (default: 500)")
    parser.add_argument("--min-elo", type=int, default=None, help="Optional minimum Elo rating filter")

    args = parser.parse_args()

    if args.pgn:
        input_paths = [Path(args.pgn)]
    else:
        input_paths = sorted(list(DATA_DIR.glob("*.pgn")))

    print(f"Target PGN files: {[p.name for p in input_paths]}")
    create_memmap_dataset(input_paths, max_games=args.max_games, min_elo=args.min_elo)

    dataset = ChessMemmapDataset()
    print(f"Loaded ChessMemmapDataset: {len(dataset):,} samples.")
    sample_x, sample_y = dataset[0]
    print(f"Verified item 0: shape {sample_x.shape}, label {sample_y.item()}")
