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
from torch.utils.tensorboard import SummaryWriter

# print cuda info
print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Cuda device count: {torch.cuda.device_count()}")
print(f"Cuda device used: {torch.cuda.get_device_name(torch.cuda.current_device())}")

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
    encoding = [0 for _ in range(len(labels))]
    labels_df = annotations_frame.loc[annotations_frame['id'] == id]
    for label, content in labels_df.items():
        if(label != 'id'):
            encoding[labels["%s-%s" % (label, labels_df[label].iloc[0])]] += 1
    return encoding


# %%
class PeopleDataset(Dataset):
    """People with annotations dataset."""

    def __init__(self, frame_with_labels, root_dir, labels, train, transform=None):
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
        self.img_files = [f for f in listdir(root_dir)]
        self.labels = labels
        self.train = train
    def __len__(self):
        return len(self.annotations_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if train:
            img_name = os.path.join(self.root_dir,self.annotations_frame.iloc[idx, 0])
            image = io.imread(img_name)
            image = T.ToTensor()(image)
            image = F.interpolate(image, size=128)  
            encoding = getTargetEncoding(self.annotations_frame.iloc[idx, 0],self.annotations_frame, self.labels)
            sample = (image, torch.tensor(encoding))
            return sample
        else:
            image = io.imread(self.img_files[idx])
            image = T.ToTensor()(image)
            image = F.interpolate(image, size=128)  
            sample = image
            return sample


# %%
def convertAnnotationsFrame(annotations_frame, train_dir):
    annotations_frame = pd.read_csv('dataset/annotations_train.csv')

    img_files = [f for f in listdir(train_dir)]

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
    return augmented_annotations_frame


# %%
#labels = setupLabelsDict(augmented_annotations_frame)
#print(augmented_annotations_frame.head())
#id = augmented_annotations_frame.iloc[1, 0]
#print(id)
#labels_df = augmented_annotations_frame.loc[augmented_annotations_frame['id'] == id]
#print(labels_df)
#encoding = getTargetEncoding(augmented_annotations_frame.iloc[1, 0],augmented_annotations_frame, labels)
#print(encoding)


# %%
#labels = setupLabelsDict(annotations_frame)
#print(len(labels))
#print(labels)
#people_dataset = PeopleDataset(frame_with_labels=augmented_annotations_frame,
#                                    root_dir='./dataset/train',
#                                    labels=labels)
#
#print("Dataset Initialized")
#dataloader = DataLoader(people_dataset, batch_size=1,
#                        shuffle=True, num_workers=0)
#print("DataLoader Initialized")
#print(len(people_dataset))


# %%
#for batch_idx, (inputs, targets) in enumerate(dataloader):
#    print(batch_idx, type(inputs), targets)
#    if batch_idx == 3:
#        break

# %% [markdown]
# # Network
# ## Fine tuning AlexNet

# %%
'''
Input arguments
  num_classes: number of classes in the dataset.
               This is equal to the number of output neurons.
'''

def initialize_alexnet(num_classes):
  # load the pre-trained Alexnet
  alexnet = torchvision.models.alexnet(pretrained=True)
  
  # get the number of neurons in the penultimate layer
  in_features = alexnet.classifier[6].in_features
  
  # re-initalize the output layer
  alexnet.classifier[6] = torch.nn.Sequential(
    torch.nn.Linear(in_features=in_features, out_features=num_classes),
    torch.nn.Sigmoid()
  )
  return alexnet

# %% [markdown]
# Cost function

# %%
def get_cost_function():
  cost_function = torch.nn.BCELoss()
  return cost_function

# %% [markdown]
# Optimizer

# %%
def get_optimizer(net, lr):
  optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.5, 0.999))
  return optimizer


# %%
def test(net, data_loader, cost_function, num_classes, device='cuda:0'):
  samples = 0.
  cumulative_loss = 0.
  cumulative_accuracy = 0.

  net.eval() # Strictly needed if network contains layers which has different behaviours between train and test
  with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(data_loader):
      print(batch_idx)
      # Load data into GPU
      inputs = inputs.to(device)
      targets = targets.to(torch.float32) #converting to float for BCELoss
      targets = targets.to(device)
      #print(inputs.size())
      #print(input)
      #print(targets.size())
      #print(targets)
        
      # Forward pass
      outputs = net(inputs)
      #print(outputs)
      # Apply the loss
      loss = cost_function(outputs, targets)

      # Better print something
      samples+=inputs.shape[0]
      cumulative_loss += loss.item() # Note: the .item() is needed to extract scalars from tensors
      predicted = torch.round(outputs)
      cumulative_accuracy += predicted.eq(targets).sum().item()/num_classes

  return cumulative_loss/samples, cumulative_accuracy/samples*100


def train(net,data_loader,optimizer,cost_function, num_classes, device='cuda:0'):
  samples = 0.
  cumulative_loss = 0.
  cumulative_accuracy = 0.

  
  net.train() # Strictly needed if network contains layers which has different behaviours between train and test
  for batch_idx, (inputs, targets) in enumerate(data_loader):
    # Load data into GPU
    inputs = inputs.to(device)
    targets = targets.to(torch.float32) #converting to float for BCELoss
    targets = targets.to(device)

    # Forward pass
    outputs = net(inputs)

    # Apply the loss
    loss = cost_function(outputs,targets)
      
    # Backward pass
    loss.backward()
    
    # Update parameters
    optimizer.step()
    
    # Reset the optimizer
    optimizer.zero_grad()

    # Better print something, no?
    samples+=inputs.shape[0]
    cumulative_loss += loss.item()
    predicted = torch.round(outputs)
    cumulative_accuracy += predicted.eq(targets).sum().item()/num_classes

  return cumulative_loss/samples, cumulative_accuracy/samples*100


# %%
def get_data(augmented_annotations_frame, labels, batch_size, img_root, test_batch_size=256):
  
  # Prepare data transformations and then combine them sequentially
  # transform = list()
  # transform.append(T.ToTensor())                            # converts Numpy to Pytorch Tensor
  # transform.append(T.Normalize(mean=[0.5], std=[0.5]))      # Normalizes the Tensors between [-1, 1]
  # transform = T.Compose(transform)                          # Composes the above transformations into one.

  # Load data
  full_training_data = PeopleDataset(frame_with_labels=augmented_annotations_frame,
                                      root_dir="%s/train" % (img_root),
                                      labels=labels,
                                      train=True)
  test_data = PeopleDataset(frame_with_labels=augmented_annotations_frame,
                                      root_dir="%s/test" % (img_root),
                                      labels=labels,
                                      train=False)

  #print("Dataset Initialized")
  #dataloader = DataLoader(people_dataset, batch_size=,
  #                        shuffle=True, num_workers=0)
  #print("DataLoader Initialized")
  #print(len(people_dataset))
  #full_training_data = torchvision.datasets.MNIST('./dataset', train=True, transform=transform, download=True) 
  #test_data = torchvision.datasets.MNIST('./dataset', train=False, transform=transform, download=True) 
  

  # Create train and validation splits
  num_samples = len(full_training_data)
  training_samples = int(num_samples*0.5+1)
  validation_samples = num_samples - training_samples

  training_data, validation_data = torch.utils.data.random_split(full_training_data, [training_samples, validation_samples])
  
  # Initialize dataloaders
  train_loader = torch.utils.data.DataLoader(training_data, batch_size, shuffle=True, num_workers=0)       #before num_workers=4
  val_loader = torch.utils.data.DataLoader(validation_data, test_batch_size, shuffle=False, num_workers=0) #before num_workers=4
  test_loader = torch.utils.data.DataLoader(test_data, test_batch_size, shuffle=False, num_workers=0) #before num_workers=4
  
  return train_loader, val_loader, test_loader


# %%
def main(batch_size=2048, 
         device='cuda:0', 
         learning_rate=0.001, 
         epochs=10, 
         img_root='./dataset'):
  
  writer = SummaryWriter(log_dir="runs/exp1")

  annotations_frame = pd.read_csv("./dataset/annotations_train.csv")
  augmented_annotations_frame = convertAnnotationsFrame(annotations_frame, "%s/train" % (img_root))
  labels = setupLabelsDict(augmented_annotations_frame)

  # Instantiates dataloaders
  train_loader, val_loader, test_loader = get_data(augmented_annotations_frame=augmented_annotations_frame, labels=labels, batch_size=batch_size, img_root=img_root)
  
  # Instantiates the model
  net = initialize_alexnet(num_classes=len(labels)).to(device)
  
  # Instantiates the optimizer
  optimizer = get_optimizer(net, learning_rate)
  
  # Instantiates the cost function
  cost_function = get_cost_function()

  print('Before training:')
  train_loss, train_accuracy = test(net, train_loader, cost_function, num_classes=len(labels))
  val_loss, val_accuracy = test(net, val_loader, cost_function, num_classes=len(labels))
  #test_loss, test_accuracy = test(net, test_loader, cost_function)

  print('\t Training loss {:.5f}, Training accuracy {:.2f}'.format(train_loss, train_accuracy))
  print('\t Validation loss {:.5f}, Validation accuracy {:.2f}'.format(val_loss, val_accuracy))
  #print('\t Test loss {:.5f}, Test accuracy {:.2f}'.format(test_loss, test_accuracy))
  print('-----------------------------------------------------')

  for e in range(epochs):
    train_loss, train_accuracy = train(net, train_loader, optimizer, cost_function, num_classes=len(labels))
    val_loss, val_accuracy = test(net, val_loader, cost_function, num_classes=len(labels))
    print('Epoch: {:d}'.format(e+1))
    print('\t Training loss {:.5f}, Training accuracy {:.2f}'.format(train_loss, train_accuracy))
    print('\t Validation loss {:.5f}, Validation accuracy {:.2f}'.format(val_loss, val_accuracy))
    print('-----------------------------------------------------')
  print('After training:')
  train_loss, train_accuracy = test(net, train_loader, cost_function, num_classes=len(labels))
  val_loss, val_accuracy = test(net, val_loader, cost_function, num_classes=len(labels))
  #test_loss, test_accuracy = test(net, test_loader, cost_function)

  print('\t Training loss {:.5f}, Training accuracy {:.2f}'.format(train_loss, train_accuracy))
  print('\t Validation loss {:.5f}, Validation accuracy {:.2f}'.format(val_loss, val_accuracy))
  #print('\t Test loss {:.5f}, Test accuracy {:.2f}'.format(test_loss, test_accuracy))
  print('-----------------------------------------------------')


# %%
# Free GPU memory
#torch.cuda.empty_cache()
#import gc
#gc.collect()


# %%
main()


