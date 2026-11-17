
import torch
import matplotlib.pyplot as plt
import numpy as np

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import LeNet5


# 1. Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 2. Class names
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
test_dataset = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=False,
    transform=transforms.ToTensor()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False
)


# 4. Load trained model
model = LeNet5().to(device)

model.load_state_dict(
    torch.load("lenet5_fashion_mnist.pth", map_location=device)
)

model.eval()


# 5. Create confusion matrix
confusion_matrix = torch.zeros(10, 10, dtype=torch.long)

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        # Count actual vs predicted labels
        indices = labels * 10 + predictions

        batch_matrix = torch.bincount(
            indices,
            minlength=100
        ).reshape(10, 10)

        confusion_matrix += batch_matrix.cpu()


# 6. Calculate per-class accuracy
print("\nPER-CLASS ACCURACY")
print("----------------------------------")

class_accuracies = []

for i, class_name in enumerate(classes):

    correct = confusion_matrix[i, i].item()
    total = confusion_matrix[i].sum().item()

    accuracy = 100 * correct / total

    class_accuracies.append(accuracy)

    print(f"{class_name:15s}: {accuracy:.2f}%")

print("----------------------------------")


# 7. Calculate overall accuracy
overall_accuracy = (
    confusion_matrix.diag().sum().item()
    / confusion_matrix.sum().item()
) * 100

print(f"Overall Accuracy: {overall_accuracy:.2f}%")


# 8. Visualize confusion matrix
cm = confusion_matrix.numpy()

plt.figure(figsize=(12, 10))

plt.imshow(cm, interpolation="nearest", cmap="Blues")
plt.title("Fashion-MNIST Confusion Matrix")
plt.colorbar()

tick_marks = np.arange(len(classes))

plt.xticks(tick_marks, classes, rotation=45, ha="right")
plt.yticks(tick_marks, classes)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

# Display values inside each cell
threshold = cm.max() / 2

for i in range(10):
    for j in range(10):

        plt.text(
            j, i, str(cm[i, j]),
            ha="center",
            va="center",
            color="white" if cm[i, j] > threshold else "black"
        )

plt.tight_layout()

plt.savefig("confusion_matrix.png", dpi=150)

print("\nConfusion matrix saved as confusion_matrix.png")


# 9. Visualize per-class accuracy
plt.figure(figsize=(12, 6))

plt.bar(classes, class_accuracies)

plt.title("Per-Class Accuracy")
plt.xlabel("Fashion Category")
plt.ylabel("Accuracy (%)")

plt.xticks(rotation=35, ha="right")
plt.ylim(0, 100)

plt.tight_layout()

plt.savefig("class_accuracy.png", dpi=150)

print("Class accuracy chart saved as class_accuracy.png")
