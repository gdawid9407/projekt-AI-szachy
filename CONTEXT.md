# PROJECT CONTEXT: Chess AI

## 1. Project Overview and Goals
- **Objective:** Build a chess engine powered by a neural network from scratch using Python and PyTorch.
- **Core Principle:** The neural network never generates illegal moves. The `python-chess` library strictly manages the game state, validates rules, and generates legal moves.
- **Primary Architecture:** Value Network (evaluating board states resulting from candidate legal moves) using a Convolutional Neural Network (CNN) / Residual architecture.
- **Learning Purpose:** Educational project focused on ML engineering, tensor representations, custom training loops, and hardware resource management.

## 2. Hardware and Infrastructure Setup
- **Development & Prototyping:** MacBook M2 (16 GB Unified Memory, macOS, Apple Silicon Metal `mps`).
- **Heavy Training (Fallback):** PC (Windows 11, NVIDIA RTX 3070 8 GB VRAM, 32 GB RAM, CUDA `cuda`).
- **Data Storage:** External SSD for raw PGN files, processed binary datasets (`np.memmap`), and saved model weights (`models/`).
- **Memory Safety:** Streaming data loading to prevent RAM exhaustion on the 16 GB machine.

## 3. Repository Structure and Implementation Status
- `[x]` `AGENT_INSTRUCTIONS.md` - System instructions and interaction rules for the AI tutor.
- `[x]` `CONTEXT.md` - Current project state and operational roadmap.
- `[x]` `requirements.txt` - Python dependencies (`torch`, `chess`, `numpy`, `tqdm`).
- `[x]` `.gitignore` - Exclusion rules for virtual environments, caches, and external datasets.
- `[x]` `src/config.py` - Global paths (`pathlib`), automatic directory creation on SSD, hyperparameters, and device detection (`cuda` -> `mps` -> `cpu`).
- `[x]` `src/encoding.py` - Converter from `chess.Board` to a `14x8x8` tensor.
- `[ ]` `src/network.py` - PyTorch neural network definition (`nn.Module`). Scaffolding initialized; `forward()` implementation pending.
- `[ ]` `src/agent.py` - Move selection logic evaluating legal positions.
- `[ ]` `play.py` - CLI script to play a test match against the model.
- `[ ]` `src/train.py` - Training pipeline (supervised learning / self-play).

## 4. Technical Constraints and Rules
- Code must use `pathlib.Path` for all paths (cross-platform compatibility).
- Dynamic hardware acceleration selection via `torch.device`.
- Explicit typing (`typing`) and modular architecture across all scripts.
- Memory handling must use chunked reading or memory mapping for large datasets.

## 5. Current Task
Complete `src/network.py`:
1. Implement forward pass logic in `ChessValueNetwork.forward`.
2. Connect feature extraction layers (`conv1` -> `bn1` -> `relu` -> `conv2` -> `bn2` -> `relu` -> `conv3` -> `bn3` -> `relu`).
3. Flatten 4D tensor `(B, 64, 8, 8)` to 2D tensor `(B, 4096)`.
4. Apply value head layers (`fc1` -> `relu` -> `fc2` -> `tanh`) producing output shape `(B, 1)`.
5. Verify output scalar evaluation on Apple Silicon MPS device.

## 6. Blockers and Open Questions
- None. Scaffolding verified with synthetic batch execution.