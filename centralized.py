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
from distributions import distribution_profiles_3_nodes, distribution_profiles_6_nodes, dataframes_from_distribution

from scipy.stats import beta
import pandas as pd

from distributions import dataset_from_distribution
from utils import dwood, save_csv_prediction, generate_project_name, get_pt_file_path, load_andrei_model_paths, \
  training_dataset

warnings.filterwarnings("ignore")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_epoch(net, trainloader, criterion, optimizer):
  net.train()
  train_loss = 0.0
  train_count = 0
  for data in tqdm(trainloader, leave=False):
    im, age, _ = data
    im = im.to(device=DEVICE, dtype=torch.float)
    age = age.to(device=DEVICE, dtype=torch.float).reshape(-1, 1)
    optimizer.zero_grad()
    pred_age = net(im)
    loss = criterion(pred_age, age)
    loss.backward()
    optimizer.step()
    train_count += im.shape[0]
    train_loss += loss.sum().detach().item()
  train_loss /= train_count
  return train_loss

def update_loss_df(losses_save_path, training_round, epoch, train_loss, val_loss):
  df = pd.read_csv(losses_save_path)
  now = datetime.datetime.now().strftime('%d-%m-%y-%H_%M')
  new_row = pd.DataFrame({
    'server_round': [training_round],
    'epoch': [epoch],
    'train_loss': [train_loss],
    'val_loss': [val_loss],
    'time': [now]})
  df = pd.concat([df, new_row], ignore_index=True)
  df.to_csv(losses_save_path, index=False)

def check_improvement(val_loss, best_loss, net, model_save_path, epoch, num_bad_epochs):
  if val_loss < best_loss:
    print(f"Epoch {epoch + 1} found new best model - saved in {model_save_path}...")
    torch.save(net.state_dict(), model_save_path)
    return True, val_loss, 0  # is_new_best, best_loss, num_bad_epochs
  else:
    return False, best_loss, num_bad_epochs + 1

def train(net, trainloader, valloader, epochs, model_save_path, losses_save_path, training_round=0, patience=5, best_loss=1e9, current_epoch=0):
  criterion = nn.L1Loss()
  optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
  #Requires the epochs to be higher than the patience for this to work
  scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience)

  num_bad_epochs = 0
  is_new_best = False

  for epoch in range(epochs):
    if num_bad_epochs >= patience:
      print(f"Model reached patience: {patience}")
      # break
    train_loss = train_epoch(net, trainloader, criterion, optimizer)
    val_loss, corr, true_ages, pred_ages, ids_sub, mae = validate(net, valloader)
    update_loss_df(losses_save_path, training_round, epoch + current_epoch, train_loss, val_loss)
    is_new_best, best_loss, num_bad_epochs = check_improvement(val_loss, best_loss, net, model_save_path, epoch, num_bad_epochs)
    scheduler.step(val_loss)
    lr = optimizer.param_groups[0]['lr']
    print(
      f'Epoch: {epoch + current_epoch + 1} of {epochs + current_epoch}, lr: {lr:.2E}, train loss: {train_loss:.2f}, valid loss: {val_loss:.2f}, corr: {corr:.2f}, best loss {best_loss:.2f}, number of epochs without improvement: {num_bad_epochs}')

  return is_new_best, best_loss, model_save_path


#Write a function to train the model
# def train(net, trainloader, valloader, epochs, model_save_path, losses_save_path, training_round=0, patience=5):
#   criterion = nn.L1Loss()
#   optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
#   scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience)
#   is_new_best = False
#   best_loss = 1e9
#   num_bad_epochs = 0
#   for epoch in range(epochs):
#     train_loss = 0.0
#     net.train()
#     train_count = 0
#     if num_bad_epochs >= patience:
#       print("model reached patience: " + str(patience))
#       return is_new_best, model_save_path
#     for i, data in enumerate(tqdm(trainloader, leave=False)):
#       im, age, idsub = data
#       im = im.to(device=DEVICE, dtype=torch.float)
#       age = age.to(device=DEVICE, dtype=torch.float)
#       age = age.reshape(-1, 1)
#       optimizer.zero_grad()
#       pred_age = net(im)
#       loss = criterion(pred_age, age)
#       loss.backward()
#       train_count += im.shape[0]
#       train_loss += loss.sum().detach().item()
#       optimizer.step()
#     train_loss /= train_count
#     val_loss, corr, true_ages, pred_ages, ids_sub, mae = validate(net, valloader)
#     #Open losses save path as a pandas dataframe
#     df = pd.read_csv(losses_save_path)
#     #Get the current time as %d-%m-%y-%H_%M
#     now = datetime.datetime.now().strftime('%d-%m-%y-%H_%M')
#     # Create a new DataFrame with the data to append
#     new_row = pd.DataFrame({'server_round': [training_round],
#                             'epoch': [epoch],
#                             'train_loss': [train_loss],
#                             'val_loss': [val_loss],
#                             'time': [now]})
#
#     # Use concat to add the new row to the existing DataFrame
#     df = pd.concat([df, new_row], ignore_index=True)
#     #Save the dataframe to the losses save path
#     df.to_csv(losses_save_path, index=False)
#     scheduler.step(val_loss)
#     if val_loss < best_loss:
#       is_new_best = True
#       best_loss = val_loss
#       print("Epoch " + str(epoch + 1) + " found new best model - saved in " + model_save_path + "...")
#       torch.save(net.state_dict(), model_save_path)
#       num_bad_epochs = 0
#     else:
#       num_bad_epochs += 1
#
#     lr = optimizer.param_groups[0]['lr']
#     print(
#       'Epoch: {} of {}, lr: {:.2E}, train loss: {:.2f}, valid loss: {:.2f}, corr: {:.2f}, best loss {:.2f}, number of epochs without improvement: {}'.format(
#         epoch + 1, epochs,
#         lr, train_loss, val_loss, corr, best_loss, num_bad_epochs))
#   return is_new_best, model_save_path

#Validates how good a model used
#Acts as the validator with the valloader and as the tester with the testloader
def validate(net, dataloader, device=DEVICE):
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
          im = im.to(device=device, dtype = torch.float)
          age = age.to(device=device, dtype=torch.float)
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

  def __init__(self, df, transform=None):
    self.file_frame = df
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


def get_data_transforms(augmentation='none'):
  """Generate data transformations based on the augmentation type."""
  base_transforms = [LoadImage(image_only=True, ensure_channel_first=True)]
  if augmentation == 'flip':
    base_transforms.append(RandFlip(prob=0.5, spatial_axis=0))
  base_transforms.append(ToTensor())
  return Compose(base_transforms)


def split_dataset_ids(df, kcrossval=None, icross=-1, test_size=0.15, random_seed=10, dataset_scale=1.0):
  """Split dataset IDs for training and validation sets, with an option to scale the dataset size."""
  IDs = df['ID'].unique().tolist()
  # Scale the dataset by selecting a subset of IDs based on the dataset_scale parameter
  scaled_size = int(len(IDs) * dataset_scale)
  np.random.seed(random_seed)  # Ensure reproducibility
  scaled_IDs = np.random.choice(IDs, scaled_size, replace=False).tolist()

  if kcrossval is None:
    return train_test_split(scaled_IDs, test_size=test_size, random_state=random_seed)
  else:
    fold_size = len(scaled_IDs) // kcrossval
    valid_ids = scaled_IDs[icross * fold_size:(icross + 1) * fold_size]
    train_ids = [id_ for id_ in scaled_IDs if id_ not in valid_ids]
    return train_ids, valid_ids


def get_loader(df, ids, transforms, batch_size):
  """Create a DataLoader given IDs and transformations."""
  idx = df[df['ID'].isin(ids)].index.tolist()
  dataset_obj = dataset(df, transform=transforms)
  sampler = SubsetRandomSampler(idx)
  return DataLoader(dataset_obj, batch_size=batch_size, sampler=sampler), len(idx)


def get_train_valid_loader(df, batch_size=4, random_seed=10, aug='none', kcrossval=None, icross=-1, dataset_scale=1.0):
  print('Composing transformations and structuring loader datasets...')
  train_transforms = get_data_transforms(augmentation=aug)
  valid_transforms = get_data_transforms(augmentation='none')
  #No cross_validation
  train_val_ids = []
  if kcrossval is None:
    train_ids, valid_ids = split_dataset_ids(df, kcrossval, icross, random_seed=random_seed, dataset_scale=dataset_scale)
    #Add these to the list
    train_val_ids.append((train_ids, valid_ids))
  else:
    for i in range(kcrossval):
      train_ids, valid_ids = split_dataset_ids(df, kcrossval, i, random_seed=random_seed, dataset_scale=dataset_scale)
      train_val_ids.append((train_ids, valid_ids))

  train_loaders = []
  valid_loaders = []

  #Initialize the train and valid loaders
  for train_ids, valid_ids in train_val_ids:
    train_loader, num_train = get_loader(df, train_ids, train_transforms, batch_size)
    valid_loader, num_valid = get_loader(df, valid_ids, valid_transforms, batch_size)
    print(f'Number of training scans: {num_train}, valid scans: {num_valid}')
    train_loaders.append(train_loader)
    valid_loaders.append(valid_loader)


  return train_loaders, valid_loaders

#Function that creates a train loader, no validation loader
def get_train_loader(df, batch_size=4, random_seed=10, aug='none', dataset_scale=1.0):
  print('Composing transformations and structuring train loader dataset...')
  train_transforms = get_data_transforms(augmentation=aug)
  # Get all unique IDs and then scale the list according to the dataset_scale parameter
  all_train_ids = df['ID'].unique().tolist()
  scaled_size = int(len(all_train_ids) * dataset_scale)
  np.random.seed(random_seed)  # Ensure reproducibility, you can choose any seed you prefer
  train_ids = np.random.choice(all_train_ids, scaled_size, replace=False).tolist()

  train_loader, num_train = get_loader(df, train_ids, train_transforms, batch_size)

  print(f'Number of training scans: {num_train}')
  return train_loader


def get_test_loader(df, batch_size, dataset_scale=1.0):
  print('Composing transformations and structuring test loader dataset...')
  test_transforms = get_data_transforms(augmentation='none')
  # Get all unique IDs and then scale the list according to the dataset_scale parameter
  all_test_ids = df['ID'].unique().tolist()
  scaled_size = int(len(all_test_ids) * dataset_scale)
  np.random.seed(10)  # Ensure reproducibility, you can choose any seed you prefer
  test_ids = np.random.choice(all_test_ids, scaled_size, replace=False).tolist()

  test_loader, num_test = get_loader(df, test_ids, test_transforms, batch_size)

  print(f'Number of test scans: {num_test}')
  return test_loader

#Function for splitting datasets
def group_datasets(df, mode='dataset', turbulence=0.0, distribution=None, nodes=6):
  #If dataset mode, split the dataframe by the dataset column
  if distribution is None:
    distribution = {}
  if mode == 'dataset':
    # Split the dataset by the dataset column
    groups = df.groupby('dataset')
    # Dont forget to reset index
    return {name: group.reset_index(drop=True, inplace=False) for name, group in groups}
  # Number mode, split uniformly in n dataframes
  elif mode == 'distribution':
    return dataframes_from_distribution(df, distribution, nodes)
  else:
    try:
      n = int(mode)
      if n < 1:
        raise ValueError("Number of splits (mode) must be at least 1.")
    except ValueError as e:
      raise ValueError("Mode must be 'dataset' or a positive integer representing the number of splits.") from e

    # Initialize the list to hold split DataFrames
    groups = []

    if turbulence > 0:
      # Adjust split points based on the turbulence factor
      alpha = 2  # Keeping alpha constant, but you can adjust this based on your needs
      beta_param = max(1, 2 / (1 + turbulence))  # Adjust beta to control skewness
      split_points = np.cumsum(beta.rvs(alpha, beta_param, size=n - 1))
      split_points /= split_points[-1]
      split_points *= len(df)
      split_points = np.unique(split_points.astype(int))  # Ensure unique split points

      # Add a start and end point for the split ranges
      split_ranges = [0] + list(split_points) + [len(df)]

      # Use iloc to split DataFrame and retain structure
      for i in range(len(split_ranges) - 1):
        start, end = split_ranges[i], split_ranges[i + 1]
        groups.append(df.iloc[start:end])
    else:
      # Calculate the size of each split
      split_size = len(df) // n
      for i in range(n):
        start = i * split_size
        # For the last split, make sure to include the remainder
        end = (i + 1) * split_size if i < n - 1 else len(df)
        groups.append(df.iloc[start:end])
  return {i: group.reset_index(drop=True, inplace=False) for i, group in enumerate(groups)}

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

def convert_state_dict(input_path):
  # function to remove the keywork 'module' from pytorch state_dict (which occurs when model is trained using nn.DataParallel)
  new_state_dict = OrderedDict()
  state_dict = torch.load(input_path, map_location='cpu')
  for k, v in state_dict.items():
    if 'module' in k:
      name = k[7:]  # remove `module.`
    else:
      name = k
    new_state_dict[name] = v
  return new_state_dict

def load_model(model_path=None):
  # Load the model
  net = DenseNet(3, 1, 1)
  if model_path:
    state_dict = convert_state_dict(model_path)
    net.load_state_dict(state_dict, strict=True)
  return net
def run_model(project_name, epochs=10, kcrossval=10, seed=None, test_dataset=None):
  save_dir = './utils/models/' + project_name + "/"
  #If save directory does not exist, create it
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print("new directory created for " + project_name)
  #Print the device we are currently working on
  print(DEVICE)
  train_df = pd.read_csv(training_dataset)
  trainloaders, valloaders = get_train_valid_loader(train_df, batch_size=3, random_seed=10, aug='none', kcrossval=kcrossval, icross=-1)
  # Load test data
  testdf = pd.read_csv(test_dataset)
  testloader = get_test_loader(testdf, batch_size=4)
  fold_losses = []
  # dwood_seed_2 = dwood + 'seed_2.pt'
  net = load_model(seed).to(DEVICE)
  init_loss, _, _, _, _, _ = validate(net, testloader)
  for k in range(kcrossval):
    print(f'Running fold {k + 1}...')
    trainloader = trainloaders[k]
    valloader = valloaders[k]
    model_save_path = save_dir  + project_name + datetime.datetime.now().strftime(f'_fold_{k}_%d-%m-%y-%H_%M.pt')
    losses_save_path = save_dir + project_name +  f'_fold_{k}' + '_losses.csv'
    # Create a new dataframe with the following columns: server_round, epoch, train_loss, val_loss, train_mae, val_mae
    if not os.path.exists(losses_save_path):
      with open(losses_save_path, 'w') as f:
        f.write('server_round,epoch,train_loss,val_loss,time\n')
    net = load_model(seed).to(DEVICE)
    epoch_test_loss = []
    best_loss = 1e9
    for epoch in range(epochs):
      _, best_loss, save_path = train(net, trainloader, valloader, 1, model_save_path, losses_save_path, best_loss=best_loss, current_epoch=epoch)
      # Validate the model using the test data
      test_loss, _, _, _, _, val_mae = validate(net, testloader)
      print(f'Epoch {epoch + 1} - test loss: {test_loss:.2f}, val mae: {val_mae:.2f}')
      epoch_test_loss.append(test_loss)
    #Append the epoch test loss to the fold losses
    fold_losses.append(epoch_test_loss)
  #Get the average epoch test loss from the folds
  fold_losses = np.array(fold_losses)
  avg_epoch_test_loss = np.mean(fold_losses, axis=0)
  #And get the standard deviation
  std_epoch_test_loss = np.std(fold_losses, axis=0)
  #Add the init loss at the start of the avg_epoch_test_loss
  avg_epoch_test_loss = np.insert(avg_epoch_test_loss, 0, init_loss)
  #Add 0 as the standard deviation for the init loss
  std_epoch_test_loss = np.insert(std_epoch_test_loss, 0, 0)
  for epoch, (avg_loss, std_loss) in enumerate(zip(avg_epoch_test_loss, std_epoch_test_loss)):
      # Write the losses to a file in save_dir
      with open(save_dir + 'centralized_losses.txt', 'a') as f:
          f.write(f"{epoch},{avg_loss},{std_loss}\n")
  # # If the repository does not exist, create it
  # if not os.path.exists(save_dir):
  #   os.makedirs(save_dir)

  # return save_path

def test_model(project_name, test_loader, state_path=None, csv_save=True, device=DEVICE):
  net = load_model(state_path).to(device)
  test_loss, corr, true_ages, pred_ages, ids_sub, mae = validate(net, test_loader, device)
  if csv_save:
    save_dir = './utils/tests/' + project_name + "/"
    # Create the directory if it does not exist
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)
    save_csv_prediction(save_dir, project_name, true_ages, pred_ages, ids_sub)
  return test_loss, corr, mae

#A function that generates centralized losses for a project_name
#Used to fix the centralized losses when the wrong test dataset is used
#Extending to also save the MAE and save predictions of the final model
def test_federated_models(project_name, test_loader, device=DEVICE, init_state_path=None):
  print(f'Testing federated models for project {project_name}...')
  # Go to the folder containing all federated models
  model_path = os.path.join('./utils', 'models', project_name)

  # Ensure model_path exists
  if not os.path.exists(model_path):
    print(f"Directory {model_path} not found.")
    return

  # In this folder, obtain all pt files that start with federated_model
  pt_files = [f for f in os.listdir(model_path) if f.startswith('federated_model')]
  # Sort the pt files by creation date (oldest first)
  pt_files.sort(key=lambda x: os.path.getctime(os.path.join(model_path, x)))
  print(pt_files)

  centralized_losses_path = os.path.join(model_path, 'centralized_losses.txt')
  centralized_mae_path = os.path.join(model_path, 'centralized_mae.txt')
  init_model = load_model(init_state_path).to(device)
  first_loss, _, _, _, _, first_mae = validate(init_model, test_loader, device)
  # #Get the first line of the file which is a format of 0,loss
  # with open(centralized_losses_path, 'r') as f:
  #   first_line = f.readline()
  #   #Get the first loss
  #   first_loss = float(first_line.split(',')[1])
  if os.path.exists(centralized_losses_path):
    # Append timestamp to the old file name
    timestamp = datetime.datetime.now().strftime('%d-%m-%y-%H_%M')
    os.rename(centralized_losses_path, os.path.join(model_path, f'centralized_losses_old_{timestamp}.txt'))
  if os.path.exists(centralized_mae_path):
    # Append timestamp to the old file name
    timestamp = datetime.datetime.now().strftime('%d-%m-%y-%H_%M')
    os.rename(centralized_mae_path, os.path.join(model_path, f'centralized_mae_old_{timestamp}.txt'))
  # Create a new file with the first loss
  with open(centralized_losses_path, 'w') as f:
    f.write(f"0,{first_loss}\n")
  with open(centralized_mae_path, 'w') as f:
    f.write(f"0,{first_mae}\n")

  #Get the amount of server rounds (amount of pt files)
  server_rounds = len(pt_files)

  for i, pt_file in enumerate(pt_files):
    #Only save prediction for the last model
    save_prediction = i == server_rounds - 1
    # Test the model
    test_loss, corr, mae = test_model(project_name, test_loader, state_path=os.path.join(model_path, pt_file),
                                      csv_save=save_prediction, device=device)
    print(f'Model {i + 1} - test loss: {test_loss:.2f}, corr: {corr:.2f}, mae: {mae:.2f}')
    # Write the losses to a file in save_dir
    # Since we have saved the old losses, we can append to the new file
    # Overwrite centralized_losses.txt
    with open(centralized_losses_path, 'a') as f:
      f.write(f"{i+1},{test_loss}\n")
    with open(centralized_mae_path, 'a') as f:
      f.write(f"{i+1},{mae}\n")

#testing centralized models
def test_centralized_models(project_name, test_loader, device=DEVICE, init_state_path=None):
  print(f'Testing centralized models for project {project_name}...')
  # Go to the folder containing all federated models
  model_path = os.path.join('./utils', 'models', project_name)

  # Ensure model_path exists
  if not os.path.exists(model_path):
    print(f"Directory {model_path} not found.")
    return

  # In this folder, obtain all pt files that start with federated_model
  pt_files = [f for f in os.listdir(model_path) if f.startswith('centralized') and f.endswith('.pt')]
  # Sort the pt files by creation date (oldest first)
  pt_files.sort(key=lambda x: os.path.getctime(os.path.join(model_path, x)))
  print(pt_files)

  #Select a random pt file in the folder
  pt_file = np.random.choice(pt_files)
  state_path = os.path.join(model_path, pt_file)
  print(state_path)
  test_loss, corr, mae = test_model(project_name, test_loader, state_path=state_path,
                                    csv_save=True, device=device)
  print(f'Model - test loss: {test_loss:.2f}, corr: {corr:.2f}, mae: {mae:.2f}')




# ./FedProx_DWood_Distribution_Gaussian_seed_2_3_Node
# ./FedProx_RW_Distribution_Original_3_Node
# ./FedProx_DWood_Distribution_Original_seed_2_3_Node
# ./FedProx_RW_Distribution_Gaussian_3_Node
#DONE


if __name__ == '__main__':
  # split_save_datasets('patients_dataset_9573.csv')
  dwood_seed_2 = dwood + 'seed_2.pt'

  # run_model('centralized_DWood_seed_2_10_fold_kcrossval', epochs=20, kcrossval=10, seed=dwood_seed_2)
  # run_model('centralized_RW_10_fold_kcrossval_test', epochs=20, kcrossval=10)
  test_df = pd.read_csv('patients_dataset_9573_test.csv')
  test_loader = get_test_loader(test_df, batch_size=4)
  # test_centralized_models('centralized_DWood', test_loader)
  # test_centralized_models('centralized_RW', test_loader)
  # test_federated_models('FedProx_RW_Distribution_Gaussian_3_Node', test_loader)



  # test_df = pd.read_csv('patients_dataset_6326_test.csv')
  # test_loader = get_test_loader(test_df, batch_size=4, dataset_scale=1)
  # project_name = generate_project_name('FedProx', 'DWood', 'Dataset', 2)
  # model_path = get_pt_file_path(project_name)
  #
  # test_model(project_name, test_loader, state_path=model_path)
  # df = pd.read_csv('patients_dataset_6326_train.csv')
  # split_save_datasets('patients_dataset_6326.csv')
  # train_df = pd.read_csv('patients_dataset_6326_train.csv')
  # test_df = pd.read_csv('patients_dataset_6326_test.csv')
  # print(f'Training dataset size: {len(train_df)}')
  # print(f'Testing dataset size: {len(test_df)}')
  # net = load_model().to(DEVICE)
  # trainloader, valloader = get_train_valid_loader(train_df, batch_size=3, random_seed=10, aug='none', kcrossval=None, icross=-1)
  # testloader = get_test_loader(test_df, batch_size=4)
  #

  # #Group the datasets
  # dfs = group_datasets(df, 'dataset')
  # #For every dataframe
  # for i, df in enumerate(dfs):
  #   #Print the dataframes
  #   print(f'Dataframe {i} size: {len(df)}')
  #   #Print the head
  #   print(df.head())

  # downsize_data('patients_dataset_6326_test.csv', percentage=5)




