import torchvision.datasets as dset
import torchvision.transforms as transforms
import os

def download_dataset(dataroot):
    # Ensure the directory exists
    if not os.path.exists(dataroot):
        os.makedirs(dataroot)

    # Download and preprocess CIFAR-10 dataset
    dataset = dset.CIFAR10(root=dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(64),  # Resize images to 64x64
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    print(f"Dataset downloaded and saved to {dataroot}")
    return dataset

if __name__ == "__main__":
    dataroot = './data'  # You can specify a custom path here
    download_dataset(dataroot)

