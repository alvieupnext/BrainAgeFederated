from monai.networks.nets import DenseNet
from collections import OrderedDict
import torch.nn as nn
from sklearn.metrics import mean_absolute_error
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import torch
from torch.utils.data import Dataset, DataLoader
import datetime
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler
import warnings
from monai.transforms import (
    Compose,
    ToTensor,
    LoadImage,
    RandFlip,
)
import pandas as pd

warnings.filterwarnings("ignore")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Write a function to train the model
def train(net, trainloader, valloader, epochs, save_path, patience=5):
  criterion = nn.L1Loss()
  optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
  scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience)
  is_new_best = False
  best_loss = 1e9
  num_bad_epochs = 0
  for epoch in range(epochs):
    train_loss = 0.0
    net.train()
    train_count = 0
    if num_bad_epochs >= patience:
      print("model reached patience: " + str(patience))
      return is_new_best, save_path
    for i, data in enumerate(tqdm(trainloader, leave=False)):
      im, age, idsub = data
      im = im.to(device=DEVICE, dtype=torch.float)
      age = age.to(device=DEVICE, dtype=torch.float)
      age = age.reshape(-1, 1)
      optimizer.zero_grad()
      pred_age = net(im)
      loss = criterion(pred_age, age)
      loss.backward()
      train_count += im.shape[0]
      train_loss += loss.sum().detach().item()
      optimizer.step()
    train_loss /= train_count
    #TODO account for different approaches
    val_loss, corr, true_ages, pred_ages, ids_sub, mae = validate(net, valloader)
    with open('results/losses.txt', 'a+') as f:
      f.write(str(train_loss) + '\n')
    scheduler.step(val_loss)
    if val_loss < best_loss:
      is_new_best = True
      best_loss = val_loss
      print("Epoch " + str(epoch + 1) + " found new best model - saved in " + save_path + "...")
      torch.save(net.state_dict(), save_path)
      num_bad_epochs = 0
    else:
      num_bad_epochs += 1

    lr = optimizer.param_groups[0]['lr']
    print(
      'Epoch: {} of {}, lr: {:.2E}, train loss: {:.2f}, valid loss: {:.2f}, corr: {:.2f}, best loss {:.2f}, number of epochs without improvement: {}'.format(
        epoch + 1, epochs,
        lr, train_loss, val_loss, corr, best_loss, num_bad_epochs))
  return is_new_best, save_path

#Validates how good a model used
#Acts as the validator with the valloader and as the tester with the testloader
def validate(net, dataloader):
  criterion = nn.L1Loss()
  val_loss = 0.0
  val_count = 0
  true_ages = []
  pred_ages = []
  ids_sub = []
  with torch.no_grad():
      net.eval()
      for k, data in enumerate(tqdm(dataloader, leave=False)):
          im, age, ids = data
          im = im.to(device=DEVICE, dtype = torch.float)
          age = age.to(device=DEVICE, dtype=torch.float)
          age = age.reshape(-1,1)
          pred_age = net(im)
          for pred, chron_age, idsub in zip(pred_age, age, ids):
              pred_ages.append(pred.item())
              true_ages.append(chron_age.item())
              ids_sub.append(idsub)

          val_loss += criterion(pred_age, age).sum().detach().item()
          val_count += im.shape[0]

      val_losses = val_loss / val_count
      corr_mat = np.corrcoef(true_ages, pred_ages)
      mae = mean_absolute_error(true_ages, pred_ages)
      corr = corr_mat[0, 1]
      return val_losses, corr, true_ages, pred_ages, ids_sub, mae
def img_to_tensor(image_path):
  ## Stijn
  """ Prepare preprocessed image for input to net

  :param image_path: str, path to T1 NIfTI
  :return: tensor, prepared image
  """
  #Possible problem due to an older version being used 3.2.1
  img = nib.load(image_path).get_fdata()  # Load preprocessed NIFTI
  img_tensor = torch.Tensor(img)  # Convert to torch Tensor
  img_tensor = torch.unsqueeze(img_tensor, dim=0)  # Add dimension
  img_tensor = (img_tensor - torch.mean(img_tensor)) / torch.std(img_tensor)  # Normalise tensor

  return img_tensor

class dataset(Dataset):
  """Brain-age fine-tuning dataset"""

  def __init__(self, csv_file, transform=None):
    self.file_frame = pd.read_csv(csv_file)
    self.transform = transform

  def __len__(self):
    return len(self.file_frame)

  def __getitem__(self, idx):
    stack_name = self.file_frame.iloc[idx]['processed_file_name']
    tensor = self.transform(stack_name)
    tensor = (tensor - tensor.mean()) / tensor.std()
    tensor = torch.clamp(tensor, -1, 5)
    tensor = torch.reshape(tensor, (1, 130, 130, 130))  ########################
    age = self.file_frame.iloc[idx]['Age']
    idsub = self.file_frame.iloc[idx]['ID']
    return tensor, age, idsub

def get_train_valid_loader(csv_file, batch_size=4, random_seed=10, aug='none', kcrossval=None, icross=-1):
  print('composing...')
  if aug == 'none':
    train_transforms = Compose([LoadImage(image_only=True, ensure_channel_first=True), ToTensor()])
  elif aug == 'flip':
    train_transforms = Compose([LoadImage(image_only=True, ensure_channel_first=True), RandFlip(prob=0.5, spatial_axis=0), ToTensor()])

  valid_transforms = Compose([LoadImage(image_only=True, ensure_channel_first=True), ToTensor()])

  print('structuring loader datasets...')
  train_dataset = dataset(csv_file, transform=train_transforms)
  valid_dataset = dataset(csv_file, transform=valid_transforms)

  df = pd.read_csv(csv_file)
  IDs = df['ID'].unique().tolist()

  if kcrossval is None:
    print('splitting training and validation datasets...')
    train_ids, valid_ids = train_test_split(IDs, test_size=0.15, random_state=random_seed)
  else:
    i = icross
    print('splitting training and validation datasets - cross validation ' + str(icross) + '/' + str(
      kcrossval - 1) + '...')
    fold_size = len(IDs) // kcrossval
    valid_ids = IDs[i * fold_size:i * fold_size + fold_size]
    train_ids = [IDs[i] for i in range(len(IDs)) if IDs[i] not in valid_ids]
  train_idx = df[df['ID'].isin(train_ids)].index.tolist()
  valid_idx = df[df['ID'].isin(valid_ids)].index.tolist()
  train_sampler = SubsetRandomSampler(train_idx)
  valid_sampler = SubsetRandomSampler(valid_idx)
  train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
  valid_loader = DataLoader(valid_dataset, batch_size=batch_size, sampler=valid_sampler)

  print('number of training scans: {}, valid scans: {}'.format(len(train_idx), len(valid_idx)))
  return train_loader, valid_loader

def get_test_loader(csv_file, batch_size):
  print('composing...')
  test_transforms = Compose([LoadImage(image_only=True, ensure_channel_first=True), ToTensor()])
  print('structuring test loader dataset...')
  test_dataset = dataset(csv_file, transform=test_transforms)
  df = pd.read_csv(csv_file)
  test_idx = df.index.tolist()
  test_sampler = SubsetRandomSampler(test_idx)
  # Creating intsances of test dataloafrt
  test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler)
  print('number of test scans: {}'.format(len(test_idx)))
  return test_loader


def split_save_datasets(csv_name, sep='\t', test_size=0.2, random_state=10):
  # Load the dataset
  df = pd.read_csv(csv_name, sep=sep)

  # Splitting the dataset
  print(f'splitting training and test datasets ({test_size * 100}%)...')
  train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

  # Preparing filenames for train and test datasets
  base_name = os.path.splitext(csv_name)[0]  # Extract base of csv_name without extension
  csv_name_train = base_name + '_train.csv'
  csv_name_test = base_name + '_test.csv'

  # Saving the training dataset
  print(f'saving training dataset in {csv_name_train}...')
  train_df.to_csv(csv_name_train, index=False, sep=sep)

  # Saving the testing dataset
  print(f'saving testing dataset in {csv_name_test}...')
  test_df.to_csv(csv_name_test, index=False, sep=sep)

  print('--------------------------------------')
  print(f'Training dataset size: {len(train_df)} - saved in {csv_name_train}')
  print(f'Testing dataset size: {len(test_df)} - saved in {csv_name_test}')
  print('--------------------------------------')

#A function to downsize the data to a smaller size 1% of the original size
def downsize_data(csv_name, sep='\t', percentage=1):
  # Load the dataset
  df = pd.read_csv(csv_name, sep=sep)

  # Downsize the dataset
  print(f'downsizing dataset to {percentage}%...')
  df = df.sample(frac=percentage / 100)

  # Preparing filename for downsized dataset
  base_name = os.path.splitext(csv_name)[0]  # Extract base of csv_name without extension
  csv_name_downsized = base_name + f'_downsized_{percentage}.csv'

  # Saving the downsized dataset
  print(f'saving downsized dataset in {csv_name_downsized}...')
  df.to_csv(csv_name_downsized, index=False, sep=sep)

  print('--------------------------------------')
  print(f'Downsized dataset size: {len(df)} - saved in {csv_name_downsized}')
  print('--------------------------------------')

  return csv_name_downsized

def run_model(project_name, epochs=10):
  save_dir = './utils/models/' + project_name + "/"
  #If save directory does not exist, create it
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print("new directory created for " + project_name)
  csv_file = 'patients_dataset_6326_test_downsized_5.csv'
  #Print the device we are currently working on
  print(DEVICE)
  trainloader, valloader = get_train_valid_loader(csv_file, batch_size=4, random_seed=10, aug='none', kcrossval=None, icross=-1)
  model_save_path = save_dir + datetime.datetime.now().strftime('{}_%d-%m-%y-%H_%M.pt'.format(project_name))
  net = DenseNet(3, 1, 1)  # , dropout_prob=0.02)
  net = net.to(device=DEVICE)
  _, save_path = train(net, trainloader, valloader, epochs, model_save_path)
  return save_path





if __name__ == '__main__':
  # downsize_data('patients_dataset_6326_test.csv', percentage=5)
  run_model('test', epochs=10)



