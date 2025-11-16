import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

# Define dataset transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset from local directory (assumes dataset is downloaded)
dataset_path = "./MIT_Places_Urban_Subset"
dataset = ImageFolder(root=dataset_path, transform=transform)

# Split dataset into training, validation, and test sets
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

# Create data loaders
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

# Display a sample image
sample_image, sample_label = dataset[0]
plt.imshow(sample_image.permute(1, 2, 0))  # Convert tensor to image
plt.title(f"Sample Image - Class {dataset.classes[sample_label]}")
plt.axis("off")
plt.show()