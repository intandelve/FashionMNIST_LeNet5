
import torch
import torch.nn as nn


class LeNet5(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=5
        )

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=5
        )

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.activation = nn.Tanh()

    def forward(self, x):

        # Convolution + Activation + Pooling
        x = self.pool1(self.activation(self.conv1(x)))

        x = self.pool2(self.activation(self.conv2(x)))

        # Flatten
        x = torch.flatten(x, start_dim=1)

        # Fully connected layers
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))

        # Output
        x = self.fc3(x)

        return x


# Test the model
if __name__ == "__main__":

    model = LeNet5()

    sample_input = torch.randn(1, 1, 28, 28)

    output = model(sample_input)

    print("Input shape:", sample_input.shape)
    print("Output shape:", output.shape)

    print("Total parameters:",
          sum(p.numel() for p in model.parameters()))
