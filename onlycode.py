# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# [https://colab.research.google.com/github/dadebulba/DeepLearningProject/blob/main/DeepLearningProject.ipynb](https://colab.research.google.com/github/dadebulba/DeepLearningProject/blob/main/DeepLearningProject.ipynb)
# %% [markdown]
# # Deep Learning Project

# %%
# import necessary libraries
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as T
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import os
from os import listdir
from os.path import isfile, join

# print cuda info
print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Cuda device count: {torch.cuda.device_count()}")
print(f"Cuda device used: {torch.cuda.get_device_name(torch.cuda.current_device())}")

# constants
DIR_DATASET_TRAIN = "./dataset/train"

# %% [markdown]
# # Dataset Preprocessing
# Make sure to extract the zip into the 'dataset' folder

# %%
annotations_frame = pd.read_csv('dataset/annotations_train.csv')
print(annotations_frame.iloc[0, 0])


# %%
def setupLabelsDict(annotations_frame):
   
    labels = {}
    index = 0
    for i in list(annotations_frame):
        if(i != "id"):
            for j in range(min(annotations_frame[i]), max(annotations_frame[i])+1):
                labels[f"{i}-{j}"] = index
                index+=1
    return labels

def getTargetEncoding(id, annotations_frame, labels):
    print("Get target encoding triggered")
    encoding = [0 for _ in range(len(labels))]
    labels_df = annotations_frame.loc[annotations_frame['id'] == id]
    for label, content in labels_df.items():
        if(label != 'id'):
            encoding[labels["%s-%s" % (label, labels_df[label].iloc[0])]] += 1
    return encoding


# %%
class PeopleDataset(Dataset):
    """People with annotations dataset."""

    def __init__(self, frame_with_labels, root_dir, labels, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.annotations_frame = frame_with_labels
        self.root_dir = root_dir
        self.transform = transform
        self.img_files = [f for f in listdir(DIR_DATASET_TRAIN)]
        self.labels = labels
    def __len__(self):
        return len(self.annotations_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,self.annotations_frame.iloc[idx, 0])
        image = io.imread(img_name)
        encoding = getTargetEncoding(self.annotations_frame.iloc[idx, 0],self.annotations_frame, labels)
        sample = (image, encoding)

        return sample


# %%
annotations_frame = pd.read_csv('dataset/annotations_train.csv')

img_files = [f for f in listdir(DIR_DATASET_TRAIN)]

augmented_annotations_list = [] 
for entry in annotations_frame.itertuples():
    for i in img_files:
        if(int(entry[1]) == int(i.split("_")[0])):
            img_with_annotation = {
                "id": i, 
                "age": entry[2], 
                "backpack":entry[3],
                "bag":entry[4],
                "handbag":entry[5],
                "clothes":entry[6],
                "down":entry[7],
                "up":entry[8],
                "hair":entry[9],
                "hat":entry[10],
                "gender":entry[11],
                "upblack":entry[12],
                "upwhite":entry[13],
                "upred":entry[14],
                "uppurple":entry[15],
                "upyellow":entry[16],
                "upgray":entry[17],
                "upblue":entry[18],
                "upgreen":entry[19],
                "downblack":entry[20],
                "downwhite":entry[21],
                "downpink":entry[22],
                "downpurple":entry[23],
                "downyellow":entry[24],
                "downgray":entry[25],
                "downblue":entry[26],
                "downgreen":entry[27],
                "downbrown":entry[28]
            }
            augmented_annotations_list.append(img_with_annotation)

augmented_annotations_frame = pd.DataFrame(augmented_annotations_list)


# %%
labels = setupLabelsDict(annotations_frame)
people_dataset = PeopleDataset(frame_with_labels=augmented_annotations_frame,
                                    root_dir='./dataset/train',
                                    labels=labels)

print("Dataset Initialized")
dataloader = DataLoader(people_dataset, batch_size=4,
                        shuffle=True, num_workers=0)
print("DataLoader Initialized")
print(len(people_dataset))


# %%
# WARNING: this will cause infinite run time
for batch_idx, (inputs, targets) in enumerate(dataloader):
    print(batch_idx, inputs, targets)
    if batch_idx == 3:
        break
