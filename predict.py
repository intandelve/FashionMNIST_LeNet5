
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import LeNet5


# 1. Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 2. Define class names
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]


# 3. Load testing dataset
transform = transforms.ToTensor()

test_dataset = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=False,
    transform=transform
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=128,
    shuffle=False
)


# 4. Load trained model
model = LeNet5().to(device)

model.load_state_dict(
    torch.load("lenet5_fashion_mnist.pth", map_location=device)
)

model.eval()


# 5. Evaluate model
correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()


test_accuracy = 100 * correct / total

print("----------------------------------")
print("Fashion-MNIST Testing Results")
print("----------------------------------")
print("Total testing images:", total)
print("Correct predictions:", correct)
print("Incorrect predictions:", total - correct)
print(f"Test Accuracy: {test_accuracy:.2f}%")


# 6. Make predictions on 12 sample images
sample_images = torch.stack(
    [test_dataset[i][0] for i in range(12)]
).to(device)

with torch.no_grad():
    outputs = model(sample_images)
    predictions = outputs.argmax(dim=1).cpu()

# 7. Visualize predictions
fig, axes = plt.subplots(3, 4, figsize=(12, 9))

for i, ax in enumerate(axes.flat):

    image, actual_label = test_dataset[i]
    predicted_label = predictions[i].item()

    ax.imshow(image.squeeze(), cmap="gray")

    ax.set_title(
        f"Actual: {classes[actual_label]}\n"
        f"Predicted: {classes[predicted_label]}",
        color="green" if actual_label == predicted_label else "red"
    )

    ax.axis("off")

plt.tight_layout()

# Save visualization
plt.savefig("predictions.png", dpi=150)

print("\nPrediction visualization saved as predictions.png")
