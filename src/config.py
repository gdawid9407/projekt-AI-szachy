from pathlib import Path
import torch

# Base directories
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"
MODELS_DIR: Path = BASE_DIR / "models"
CHECKPOINTS_DIR: Path = BASE_DIR / "checkpoints"

# Board representation dimensions
# 14 channels x 8 ranks x 8 files:
# 6 channels for White pieces (P, N, B, R, Q, K)
# 6 channels for Black pieces (p, n, b, r, q, k)
# 1 channel for Active color turn
# 1 channel for Castling rights
BOARD_CHANNELS: int = 14
BOARD_SIZE: int = 8


def get_device() -> torch.device:
    """
    Selects the most performant compute accelerator available.
    Hierarchy: CUDA -> MPS (Apple Silicon) -> CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    # TODO: Implement Apple Silicon Metal Performance Shaders (MPS) check here.
    # Check if PyTorch MPS backend is available. If so, return torch.device("mps").
    
    return torch.device("cpu")
