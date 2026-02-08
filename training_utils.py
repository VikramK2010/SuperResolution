import os

import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader

def load_class_names(train_dir):
  # each subdir is a class
  class_names = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
  class_names.sort()
  return class_names

def load_image_files(train_dir, class_names):
  # each subdir is a class
  image_files = []
  labels = []
  for i, class_name in enumerate(class_names):
    class_dir = os.path.join(train_dir, class_name)
    files = [os.path.join(class_dir, f) for f in os.listdir(class_dir)]
    image_files.extend(files)
    labels.extend([i] * len(files))
  return image_files, labels

def build_plant_dict(class_names):
   # {plant_name: [class_names,...],...}
    plant_dict = defaultdict(list)
    for class_name in class_names:
        plant_name = class_name.split('_')[0]  # assuming class names are like 'PlantA_1', 'PlantA_2', etc.
        plant_dict[plant_name].append(class_name)
    return plant_dict

def load_plant_balanced_image_file_sample(train_dir, plant_dict, min_count=1000):
  # each subdir is a class
  image_files = []
  labels = []
  class_names = []
  # find the minimum number of classes per plant
  # min_count = min(len(classes) for classes in plant_dict.values())
  max_count = max(len(classes) for classes in plant_dict.values())
  max_images_per_class = min_count * max_count 
  print(f"Balancing plants with min classes: {min_count}, max classes: {max_count}")

  for plant_name, classes in plant_dict.items():
    for i, class_name in enumerate(classes):
      class_dir = os.path.join(train_dir, class_name)
      files = [os.path.join(class_dir, f) for f in os.listdir(class_dir)]
      sampled_files = np.random.choice(files, size=min(max_images_per_class // len(classes), len(files)), replace=False)
      image_files.extend(sampled_files)
      labels.extend([i] * len(sampled_files))
      class_names.append(class_name)
  return image_files, labels, class_names

def plot_image(image_file):
  img = plt.imread(image_file)
  plt.imshow(img)
  plt.axis('off')
  plt.show()

def graph_class_examples(train_dir, class_name, num_examples=2):
  class_dir = os.path.join(train_dir, class_name)
  files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) ]
  files = files[:num_examples]
  plt.figure(figsize=(15,5))
  for i, file in enumerate(files):
    img = plt.imread(file)
    plt.subplot(1, num_examples, i + 1)
    plt.imshow(img)
    plt.axis('off')
  plt.show()

def data_summary(image_files, labels, class_names):
  label_counts = Counter(labels)
  print("Dataset Summary:")
  print(f"Total images: {len(image_files)}")
  print(f"Total classes: {len(class_names)}")
  for i, class_name in enumerate(class_names):
    print(f"Class '{class_name}': {label_counts[i]} images")

def summarize_image_attributes(image_files, sample_size=100):
  widths = []
  heights = []
  # sample n random images
  sampled_files = np.random.choice(image_files, size=min(sample_size, len(image_files)), replace=False)
  for file in sampled_files:
    img = plt.imread(file)
    heights.append(img.shape[0])
    widths.append(img.shape[1])
  print(f"Image widths: min={min(widths)}, mean={np.mean(widths)}, max={max(widths)}, std={np.std(widths)}")
  print(f"Image heights: min={min(heights)}, mean={np.mean(heights)}, max={max(heights)}, std={np.std(heights)}")

def load_image_file_sample(train_dir, class_names, min_count=1000):
  # each subdir is a class
  image_files = []
  labels = []
  for i, class_name in enumerate(class_names):
    class_dir = os.path.join(train_dir, class_name)
    files = [os.path.join(class_dir, f) for f in os.listdir(class_dir)]
    sampled_files = np.random.choice(files, size=min(min_count, len(files)), replace=False)
    image_files.extend(sampled_files)
    labels.extend([i] * len(sampled_files))
  return image_files, labels

def visualize_results_rgb(generator, test_loader, device, epoch):
    generator.eval()
    with torch.no_grad():
        for low_res, high_res in test_loader:
            low_res, high_res = low_res.to(device), high_res.to(device)
            fake_high_res = generator(low_res)

            # Show first 4 images
            fig, axes = plt.subplots(3, 4, figsize=(12, 9))
            for i in range(4):
                def to_rgb_image(tensor):
                    img = tensor.detach().cpu().clamp(0, 1)
                    # if img.min() < 0:
                    #     img (img + 1) / 2  # rescale to [0, 1]
                    img = img.permute(1, 2, 0).numpy()  # C x H x W to H x W x C
                    return img 

                # Low resolution
                # Error low_res image is a (3 x 64 x 64) tensor
                axes[0, i].imshow(to_rgb_image(low_res[i]))
                axes[0, i].set_title("Low Res")
                axes[0, i].axis("off")

                # Generated high resolution
                axes[1, i].imshow(to_rgb_image(fake_high_res[i]))
                axes[1, i].set_title("Generated")
                axes[1, i].axis("off")

                # Real high resolution
                axes[2, i].imshow(to_rgb_image(high_res[i]))
                axes[2, i].set_title("Real High Res")
                axes[2, i].axis("off")

            plt.suptitle(f"Epoch {epoch}")
            plt.tight_layout()
            plt.savefig(f"results_epoch_{epoch}.png")
            plt.show()
            break
    generator.train()

def visualize_results(generator, test_loader, device, epoch):
    generator.eval()
    with torch.no_grad():
        for low_res, high_res in test_loader:
            low_res, high_res = low_res.to(device), high_res.to(device)
            fake_high_res = generator(low_res)

            # Show first 4 images
            fig, axes = plt.subplots(3, 4, figsize=(12, 9))
            for i in range(4):
                # Low resolution
                # Error low_res image is a (3 x 64 x 64) tensor
                axes[0, i].imshow(low_res[i].cpu().squeeze(), cmap="gray")
                axes[0, i].set_title("Low Res")
                axes[0, i].axis("off")

                # Generated high resolution
                axes[1, i].imshow(fake_high_res[i].cpu().squeeze(), cmap="gray")
                axes[1, i].set_title("Generated")
                axes[1, i].axis("off")

                # Real high resolution
                axes[2, i].imshow(high_res[i].cpu().squeeze(), cmap="gray")
                axes[2, i].set_title("Real High Res")
                axes[2, i].axis("off")

            plt.suptitle(f"Epoch {epoch}")
            plt.tight_layout()
            plt.savefig(f"results_epoch_{epoch}.png")
            plt.show()
            break
    generator.train()


