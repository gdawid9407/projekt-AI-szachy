import torch
import torch.nn as nn
from src.config import BOARD_CHANNELS, BOARD_SIZE


class ChessValueNetwork(nn.Module):
    """
    Convolutional Value Network for chess position evaluation.
    Maps an encoded board state tensor of shape (B, 14, 8, 8) to a scalar
    evaluation value v in the range [-1.0, 1.0].
    """

    def __init__(self, num_filters: int = 64) -> None:
        super().__init__()

        # Feature extraction layers
        # Input: (B, 14, 8, 8) -> Conv/BatchNorm/ReLU blocks preserving 8x8 spatial dimensions
        self.conv1 = nn.Conv2d(BOARD_CHANNELS, num_filters, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)

        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)

        self.conv3 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(num_filters)

        self.relu = nn.ReLU()

        # Dimension calculation for fully connected layer:
        # Spatial dimensions remain 8x8 through padding=1 convolutions.
        flattened_dim = num_filters * BOARD_SIZE * BOARD_SIZE

        # Value evaluation head
        # TODO: Define the dense layers leading to the scalar evaluation.
        # Layer requirements:
        # 1. Linear layer from flattened_dim to 128 hidden units.
        # 2. Linear layer from 128 to 1 unit.
        # 3. Tanh activation to bound the final evaluation in [-1.0, 1.0].
        self.fc1 = nn.Linear(flattened_dim, 128)
        self.fc2 = nn.Linear(128, 1)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (B, 14, 8, 8).
        Returns:
            torch.Tensor: Scalar value evaluation of shape (B, 1).
        """
        # Feature extraction: (B, 14, 8, 8) -> (B, 64, 8, 8)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))

        # Flatten spatial dimensions preserving batch dimension: (B, 64, 8, 8) -> (B, 4096)
        x = torch.flatten(x, start_dim=1)

        # Value evaluation head: (B, 4096) -> (B, 128) -> (B, 1)
        x = self.relu(self.fc1(x))
        x = self.tanh(self.fc2(x))

        return x


if __name__ == "__main__":
    from src.config import get_device

    device = get_device()
    model = ChessValueNetwork().to(device)

    # Synthetic batch of 2 encoded boards: (batch_size, channels, ranks, files)
    dummy_input = torch.randn(2, BOARD_CHANNELS, BOARD_SIZE, BOARD_SIZE, device=device)

    try:
        output = model(dummy_input)
        print(f"Model forward pass successful. Output shape: {output.shape}")  # Expected: torch.Size([2, 1])
        print(f"Sample evaluations: {output.squeeze(-1).tolist()}")
    except NotImplementedError as e:
        print(f"Scaffolding initialized. Pending completion: {e}")
