import matplotlib
matplotlib.use("TkAgg")
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 1. Define image transformation
transform = transforms.ToTensor()

# 2. Download training dataset
train_dataset = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

# 3. Download testing dataset
test_dataset = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# 4. Display dataset information
print("Dataset downloaded successfully!")
print("----------------------------------")
print("Training images:", len(train_dataset))
print("Testing images:", len(test_dataset))

# 5. Check one sample image
image, label = train_dataset[0]

print("Image shape:", image.shape)
print("Label:", label)

# 6. Display image
plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"Label: {label}")
plt.axis("off")
plt.show()
