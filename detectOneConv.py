import torch
from torch import nn
from torchvision.transforms import ToTensor, Resize, Grayscale, Compose, Normalize
from PIL import Image
import sys
import numpy as np
import matplotlib.pyplot as plt

classes = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc1 = nn.Linear(32*14*14, 600)
        self.fc2 = nn.Linear(600, 120)
        self.fc3 = nn.Linear(120, 60)
        self.fc4 = nn.Linear(60, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)

        return out

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Użycie: python3 detect.py <ścieżka_do_obrazka>")
        sys.exit(1)

    image_path = sys.argv[1]
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    try:
        transform = Compose([
            Grayscale(),
            Resize((28, 28)),
            ToTensor(),
            Normalize((0.1307,), (0.3081,))
        ])

        image = Image.open(image_path)
        original_image = image.copy()

        image = transform(image).unsqueeze(0).to(device)

        model = ConvNet(10).to(device)
        model.load_state_dict(torch.load('oneConv.pth'))
        model.eval()

        with torch.no_grad():
            logits = model(image)
            probabilities = torch.softmax(logits, dim=1)[0]
            predicted_class = logits.argmax(1).item()

        print(f"Wykryto: {classes[predicted_class]}")

        bars = classes
        y_pos = np.arange(len(probabilities.cpu()))
        plt.figure(figsize=(10, 5))

        probabilities_cpu = probabilities.cpu().numpy()
        plt.bar(y_pos, probabilities_cpu, color='#969696')
        plt.xticks(y_pos, bars)
        plt.xlabel('Klasa', fontsize=12, color='#323232')
        plt.ylabel('Przewidywanie', fontsize=12, color='#323232')

        plt.figure(figsize=(5, 5))
        plt.imshow(original_image, cmap='gray')
        plt.title(f"Wykryto: {classes[predicted_class]} ({probabilities[predicted_class]:.2f})", fontsize=16)
        plt.axis('off')

        plt.show()

    except Exception as e:
        print(f"Wystąpił błąd: {e}")

