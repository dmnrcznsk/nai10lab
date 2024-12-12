import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import sys
import numpy as np
import matplotlib.pyplot as plt

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.flatten = nn.Flatten() 
        self.unflatten = nn.Unflatten(1, (1, 28, 28))  

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.flatten(x)  
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.unflatten(x)  
        return x

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Użycie: python3 encode.py <ścieżka_do_obrazka>")
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
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
        ])

        image = Image.open(image_path)
        original_image = image.copy()

        image = transform(image).unsqueeze(0).to(device)

        model = Autoencoder().to(device)
        model.load_state_dict(torch.load('autoencoder.pth'))  
        model.eval()

        with torch.no_grad():
            reconstructed_image = model(image)

        reconstructed_image = reconstructed_image.squeeze(0).cpu().numpy().reshape(28, 28)

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(np.array(original_image), cmap='gray')
        plt.title("Oryginalne zdjęcie")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(reconstructed_image, cmap='gray')
        plt.title("Zrekonstruowane zdjęcie")
        plt.axis('off')

        plt.show()

    except Exception as e:
        print(f"Wystąpił błąd: {e}")

