import sys
import pandas as pd
import numpy as np
import os
import csv
import re
import itertools
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from generators import SimpleGenerator, ResidualGenerator, AttentionMedicalGenerator
from agricultural_sr import AgriculturalAttentionGenerator, AgriculturalDiscriminator, AgriculturalTrainer
from training_classes import AdaptiveTrainer,  MedicalTrainer, EnhancedTrainer
from discriminators import SimpleDiscriminator, SpectralNormDiscriminator
from training_loop import enhanced_training_loop
from torch.utils.data import DataLoader
import dataset_utils



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





def test_run():
    train_dir = "data/agricultural_images/train" # temporary hardcode
    class_names = load_class_names(train_dir)
    # print(class_names)
    # image_files, labels = load_image_files(train_dir, class_names)
    # graph_class_examples(train_dir, class_names[0])
    # data_summary(image_files, labels, class_names)
    # summarize_image_attributes(image_files)

    min_examples_per_class = 50 

    sample_image_files, sample_labels = load_image_file_sample(train_dir, class_names, min_examples_per_class)
    # graph_class_examples(train_dir, class_names[0])
    # data_summary(sample_image_files, sample_labels, class_names)
    # summarize_image_attributes(sample_image_files)


    dataset = dataset_utils.SuperResolutionDataset(sample_image_files)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)



    generator, discriminator = enhanced_training_loop(dataset,
                                            dataloader,
                                            AgriculturalTrainer,
                                            AgriculturalAttentionGenerator,
                                            AgriculturalDiscriminator)

def full_run():
    pass

def main():
    test_run()
    # full_run()

# Run the main function
# python example_training_run.py
if __name__ == "__main__":
    main()
