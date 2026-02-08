import glob
from torch.utils.data import Dataset

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

from PIL import Image
import matplotlib.pyplot as plt


class SuperResolutionDataset(Dataset):
    def __init__(self, image_files, low_res_size=64, high_res_size=256):
        self.image_files = image_files
        self.low_res_size = low_res_size
        self.high_res_size = high_res_size

        # Transforms for preprocessing
        self.to_tensor = transforms.ToTensor()
        # self.normalize = transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
        # Color normalization for 3 channels
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load file
        # image = plt.imread(self.image_files[idx])
        try:
            image = Image.open(self.image_files[idx])
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as e:
            print(f"Error loading image {self.image_files[idx]}: {e}")
            # Create a blank image in case of error
            image = Image.new('RGB', (self.high_res_size, self.high_res_size), (0, 0, 0))

        # Normalize to 0-255 range
        # image = (image - image.min()) / (image.max() - image.min()) * 255

        # Convert to PIL Image and resize to high resolution
        # image = transforms.ToPILImage()(image)

        #high_res_img = Image.fromarray(image).convert('L')
        high_res_img = image.resize((self.high_res_size, self.high_res_size), resample=Image.Resampling.BICUBIC)

        # Create low resolution version by downsampling
        low_res_img = image.resize((self.low_res_size, self.low_res_size), resample=Image.Resampling.BICUBIC)

        # Convert to tensors and normalize
        high_res_tensor = self.normalize(self.to_tensor(high_res_img))
        low_res_tensor = self.normalize(self.to_tensor(low_res_img))

        return low_res_tensor, high_res_tensor

# X = labels
# sample_image_files = [...]
# Example SuperResolutionDatasetWithLabels(sample_image_files, X) =
# = SuperResolutionDatasetWithLabels(image_files=sample_image_files, image_labels=X) 
class SuperResolutionDatasetWithLabels(Dataset):
    def __init__(self, image_files, image_labels, low_res_size=64, high_res_size=256):
        self.image_files = image_files
        self.image_labels = image_labels
        self.low_res_size = low_res_size
        self.high_res_size = high_res_size

        # Transforms for preprocessing
        self.to_tensor = transforms.ToTensor()
        # self.normalize = transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
        # Color normalization for 3 channels
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load file
        try:
            image = Image.open(self.image_files[idx])
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as e:
            print(f"Error loading image {self.image_files[idx]}: {e}")
            # Create a blank image in case of error
            image = Image.new('RGB', (self.high_res_size, self.high_res_size), (0, 0, 0))

        label = self.image_labels[idx]

        # Normalize to 0-255 range
        # image = (image - image.min()) / (image.max() - image.min()) * 255

        # Convert to PIL Image and resize to high resolution
        # image = transforms.ToPILImage()(image)
        
        #high_res_img = Image.fromarray(image).convert('L')
        high_res_img = image.resize((self.high_res_size, self.high_res_size), resample=Image.Resampling.BICUBIC)

        # Create low resolution version by downsampling
        low_res_img = image.resize((self.low_res_size, self.low_res_size), resample=Image.Resampling.BICUBIC)

        # Convert to tensors and normalize
        high_res_tensor = self.normalize(self.to_tensor(high_res_img))
        low_res_tensor = self.normalize(self.to_tensor(low_res_img))

        return label, low_res_tensor, high_res_tensor
