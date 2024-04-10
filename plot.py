# Based on the file content, we need to modify the function to correctly parse the losses.
# The losses are in the format "index,loss_value".
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from utils import generate_save_dir, plot_folder


#STD indicates whether the text file contains the standard deviation of the losses
def get_centralized_losses(project_name, std=False):
  folder_path = generate_save_dir(project_name)
  # Construct the full path for the 'centralized_losses.txt' file
  file_path = os.path.join(folder_path, 'centralized_losses.txt')

  # Initialize an empty array to hold the losses
  losses = []

  # Try to open the file and read the contents
  try:
    with open(file_path, 'r') as file:
      # Read each line in the file
      for line in file:
        # Split the line by comma and try to convert the second element to a float
        try:
          if std:
            index, loss, std = line.strip().split(',')
          else:
            index, loss = line.strip().split(',')
          losses.append(float(loss))
        except ValueError:
          # If a line cannot be split or converted to float, ignore it
          continue
  except FileNotFoundError:
    # If the file does not exist, print an error message
    print(f"The file {file_path} does not exist.")

  # Return the array of losses
  return losses

#Function for retrieving the losses from the different decentralized folders
def get_decentralized_losses(project_name):
  folder_path = generate_save_dir(project_name)
  # Get all folder names of the folders in this path, these are the clients
  clients = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
  # Initialize an empty array to hold the losses
  training_losses = []
  validation_losses = []
  # Hold the x-ticks for the plot
  x_ticks = []
  for client in clients:
    # Construct the full path for the f'{client}_losses.csv' file as a pandas dataframe
    file_path = os.path.join(folder_path, client, f'{client}_losses.csv')
    # Try to open the file and read the contents
    # Open the file as a pandas dataframe
    df = pd.read_csv(file_path)
    #For every row in the dataframe, append the training and validation losses and x-ticks to the arrays
    training_losses.append(df['train_loss'].values)
    validation_losses.append(df['val_loss'].values)
    #Make the server/epoch x-ticks
    df['server_epoch'] = df['server_round'].astype(str) + '/' + df['epoch'].astype(str)
    x_ticks.append(df['server_epoch'].values)
  # Return the arrays of losses and the x-ticks
  return clients, training_losses, validation_losses, x_ticks

def get_decentralized_losses_kcrossval(project_name):
  folder_path = generate_save_dir(project_name)
  # Get all folder names of the folders in this path, these are the clients
  clients = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
  # Initialize an empty array to hold the losses
  training_losses_all = []
  stds_all = []
  x_ticks_all = []
  for client in clients:
    # Initialize an empty array to hold the losses
    training_losses = []
    stds = []
    x_ticks = []
    # Construct the full path for the f'{client}_losses.csv' file as a pandas dataframe
    file_path = os.path.join(folder_path, client, f'{client}_losses.txt')
    # Try to open the file and read the contents
    # Try to open the file and read the contents
    with open(file_path, 'r') as file:
      # Read each line in the file
      for line in file:
        # Split the line by comma and try to convert the second element to a float
        index, loss, std = line.strip().split(',')
        x_ticks.append(index)
        training_losses.append(float(loss))
        stds.append(float(std))
    # Append the training losses and x-ticks to the arrays
    training_losses_all.append(training_losses)
    stds_all.append(stds)
    x_ticks_all.append(x_ticks)
  # Return the arrays of losses and the x-ticks
  return clients, training_losses_all, stds_all, x_ticks_all

def merge_client_names(client_names_array):
  final_client_names = []
  seen_client_names = []
  for client_names in client_names_array:
    #Check equality between the client_names and the final_client_names
    #If the client_name is not equal to the final_client_name, replace it by client_name if the final_client_name is empty
    #If the final_client_names is not empty, fuse the strings between the client_name and the final_client_name, seperated by a /
    if not final_client_names:
      final_client_names = client_names
      seen_client_names.append(client_names)
    elif client_names not in seen_client_names:
        final_client_names = [f'{client_name}/{final_client_name}' for client_name, final_client_name in zip(client_names, final_client_names)]
        seen_client_names.append(client_names)
  print(final_client_names)
  return final_client_names

def merge_x_ticks(xticks_array):
  #xticks array is an array that contains all xticks
  #The xticks array is a list of lists
  #Return the largest array in the list
  return max(xticks_array, key=len)

#
# # For every strategy, generate a model and for every DWood seed
def get_results(strategies, model, seeds, data, alias=None, kcrossval=False):
  project_losses = {}
  client_losses = {}
  for m in model:
    centralized_name = f'centralized_{m}'
    # if alias:
    #   centralized_name += f'_{alias}'
    loss = get_centralized_losses(centralized_name, std=True)
    project_losses[centralized_name] = loss
    for s in strategies:
      for d in data:
        strategy_data_name = f'{s}_{m}'
        # Generate the project name
        project_name = f'{strategy_data_name}_{d}'
        print(project_name)
        #Make an array of the project names
        project_names = []
        # For every seed, add the seed to the project name if the mode is DWood
        if m == 'DWood':
          for seed in seeds:
            seed_name = f'_seed_{seed}'
            if alias:
              project_names.append(project_name + seed_name + f'_{alias}')
        else:
          if alias:
            project_name += f'_{alias}'
          project_names.append(project_name)
        print(project_names)
        # Get the losses for the project
        losses = [get_centralized_losses(p) for p in project_names]
        #Get the mean loss
        losses = np.array(losses).mean(axis=0)
        # Add the project name and the losses to the dictionary
        project_losses[project_name] = losses
        if kcrossval:
          client_tuples = [get_decentralized_losses_kcrossval(p) for p in project_names]
        else:
          # Get the clients, training, validation losses and x-ticks from the project
          client_tuples = [get_decentralized_losses(p) for p in project_names]
        # Generate 4 different arrays for the clients, training, val loss/std and x-ticks
        # Val loss is only relevant for single model training
        # std is only relevant for kcrossval
        clients, training_losses, val_loss_or_std, x_ticks = zip(*client_tuples)
        #For the clients and x-ticks, take the first array in the list and use that
        clients = merge_client_names(clients)
        x_ticks = merge_x_ticks(x_ticks)
        #Take the mean of the training and validation losses
        training_losses = np.array(training_losses).mean(axis=0)
        val_loss_or_std = np.array(val_loss_or_std).mean(axis=0)
        #Add these 4 things to the client_losses dictionary
        client_losses[project_name] = (clients, training_losses, val_loss_or_std, x_ticks)
        # for project_name in project_names:
        #   losses = get_centralized_losses_corrected(project_name)
        #   # Print the project name and the losses
        #   print(f'Project: {project_name}, Losses: {losses}')
        #   # Add the project name and the losses to the dictionary
        #   project_losses[project_name] = losses
  return project_losses, client_losses

# Define the plotting function
def plot_losses(projects_losses, split='Dataset', mode='DWood', alias=None):
    # Set the aesthetic style of the plots
    sns.set(style="whitegrid")

    # Create a figure and a set of subplots
    plt.figure(figsize=(10, 8))

    # Find the maximum length of the arrays in the dataset
    max_length = max(len(losses) for losses in projects_losses.values())

    # Extend any array shorter than the max_length to match the max_length
    # Extend any array shorter than the max_length with its last element
    for project_name, losses in projects_losses.items():
      if len(losses) < max_length:
        projects_losses[project_name] = np.pad(losses, (0, max_length - len(losses)), 'edge')

    # Loop through the dictionary and plot each project's losses
    for project_name, losses in projects_losses.items():
        rounds = list(range(len(losses)))  # Server rounds
        plt.plot(rounds, losses, label=project_name)
        # Give a friendly name to the mode RW -> Random Weights
    if mode == 'RW':
      verbose_mode = 'Random Weights'
    else:
      verbose_mode = mode

    # Customize the plot
    title = f'Losses Over Server Rounds ({split}, {verbose_mode})'
    if alias:
      title += f' ({alias})'
    plt.title(title, fontsize=16)
    plt.xlabel('Server Rounds/Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(title='Strategies')
    sns.despine()
    #Save the plot at the plot folders
    file_name = f'losses_over_server_rounds_{split}_{mode}'
    if alias:
      file_name += f'_{alias}'
    path = os.path.join(plot_folder, f'{file_name}.pdf')
    #Save as plot
    plt.savefig(path, format='pdf', dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()


def plot_client_losses(client_losses, client_number=12, split='Dataset', mode='DWood', alias='', kcrossval=False, runs=10):
  # Set the visual theme of the plots with Seaborn
  sns.set(style="whitegrid")

  #Get the amouunt of clients and turn it into a rows and column
  #The amount of rows varies and the columns are always 4
  rows = int(np.ceil(client_number / 2))
  columns = 2


  # Create a 4x3 grid of subplots with shared x and y axes for uniform scale
  fig, axes = plt.subplots(rows, columns, figsize=(14,14), sharex=True, sharey=True)
  axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration

  # From client_losses, retrieve all clients
  client_tuples = list(client_losses.values())
  clients = [client_tuple[0] for client_tuple in client_tuples]
  clients = merge_client_names(clients)


  # Iterate through each project in the client_losses dictionary
  for project_name, (_, training_losses, val_loss_or_std, x_ticks) in client_losses.items():
    # Iterate through each client and corresponding subplot axis
    for idx, (client, ax) in enumerate(zip(clients, axes)):
      # Plot the training loss for the current client
      sns.lineplot(x=x_ticks[idx], y=training_losses[idx], ax=ax, label=f'{project_name} Train Loss', marker='o')
      if kcrossval:
        #Generate the 95% confidence interval
        lower_bound = np.array(training_losses[idx]) - 1.96 * np.array(val_loss_or_std[idx]) / runs ** 0.5
        upper_bound = np.array(training_losses[idx]) + 1.96 * np.array(val_loss_or_std[idx]) / runs ** 0.5
        #Plot the 95% confidence interval, use fill between
        ax.fill_between(x_ticks[idx], lower_bound, upper_bound, alpha=0.3)
      else:
        sns.lineplot(x=x_ticks[idx], y=val_loss_or_std[idx], ax=ax, label=f'{project_name} Val Loss', marker='s',linestyle=':')
      # # # Plot the validation loss for the current client with a dotted line

      # Set the title of the subplot to the client's name
      ax.set_title(f'{client}')
      # Set the labels for the x and y axes
      if kcrossval:
        ax.set_xlabel('Server Round')
      else:
        ax.set_xlabel('Server/Epoch')
      ax.set_ylabel('Loss')
      # Display the legend to differentiate between training and validation loss
      ax.legend()

  # Give a friendly name to the mode RW -> Random Weights
  if mode == 'RW':
    verbose_mode = 'Random Weights'
  else:
    verbose_mode = mode
  # Set the main title for the entire figure, positioned above all subplots
  title = f'Losses by Client ({split}, {verbose_mode})'
  if alias:
    title += f' ({alias})'
  plt.suptitle(title, fontsize=20)
  # Adjust the layout to make sure there's no overlap between subplots
  plt.tight_layout()
  file_name = f'Losses_by_client_{split}_{mode}'
  if alias:
    file_name += f'_{alias}'
  path = os.path.join(plot_folder, f'{file_name}.pdf')
  #Save the plot in the utils folder
  plt.savefig(path, format='pdf', dpi=300, bbox_inches='tight')
  # Display the plot
  plt.show()

# #If the plot folder does not exist, create it
if not os.path.exists(plot_folder):
  os.makedirs(plot_folder)

# Get the results


# Plot the losses
# plot_losses(project_losses, split='Distribution', mode='DWood')
# plot_client_losses(client_losses,4, split='Distribution', mode='DWood')
#
# # Read patients_dataset_9573_noage.csv
# dataset = pd.read_csv('patients_dataset_9573.csv')

# Plot the Age distribution, make it a function
# Add spacing between the bins
def plot_age_distribution(data, save_path=None, node=None):
  plt.figure(figsize=(10, 6))  # Increase figure size
  sns.set(style="whitegrid")  # Use seaborn style for prettier plots
  title = f'Age Distribution'
  if node:
    title += f' (Node: {node})'
  data['Age'].plot(kind='hist', bins=20, title=title, color='skyblue', edgecolor='black', rwidth=0.8)
  plt.xlabel('Age')
  plt.ylabel('Frequency')
  plt.grid(True)  # Add grid lines
  plt.tight_layout()  # Adjust layout to not cut off labels

  # Save the figure to a PDF at the specified path, if needed
  if save_path:
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
  plt.show()

# Plot the dataset_name distribution, make it a function
# Only give a top 20 of the dataset_name, all other datasets should be classified as 'Other'
def plot_dataset_distribution(data, save_path=None):
    value_counts = data['dataset_name'].value_counts()
    for dataset_name, count in value_counts.items():
      print(f'{dataset_name}: {count}')
    top_20 = value_counts.nlargest(20)
    other_count = value_counts[20:].sum()
    top_20['Other'] = other_count

    plt.figure(figsize=(12, 8))  # Increase figure size
    sns.set(style="whitegrid")  # Use seaborn style
    top_20.plot(kind='bar', title='Top 20 Dataset Distribution', color='cadetblue', edgecolor='black')
    plt.ylabel('Amount of Patients')
    plt.xlabel('Dataset Name')
    plt.xticks(rotation=45, ha='right')  # Rotate and align labels for readability
    plt.grid(axis='y')  # Add horizontal grid lines
    plt.tight_layout()  # Adjust layout
    if save_path:
      plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.show()


def plot_parent_dataset_distribution(data, save_path=None):
    plt.figure(figsize=(10, 6))  # Increase figure size
    sns.set(style="whitegrid")  # Use seaborn style
    data['dataset'].value_counts().plot(kind='bar', title='Dataset Distribution', color='lightcoral', edgecolor='black')
    value_counts = data['dataset'].value_counts()
    for dataset_name, count in value_counts.items():
      print(f'{dataset_name}: {count}')
    plt.ylabel('Amount of Patients')
    plt.xlabel('Dataset')
    plt.xticks(rotation=45, ha='right')  # Rotate and align labels
    plt.grid(axis='y')  # Add horizontal grid lines
    plt.tight_layout()  # Adjust layout
    if save_path:
      plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
  strategies = ['FedAvg', 'FedProx']
  model = ['DWood']
  seeds = [2]
  # data = ['Dataset']
  data = ['Distribution_Gaussian', 'Distribution_Original']
  alias = '3_Node'
  project_losses, client_losses = get_results(strategies, model, seeds, data, alias=alias, kcrossval=True)
  # print(project_losses.keys())
  # print(client_losses.keys())
  # # Print the results
  # # print(project_losses)
  # # print(client_losses)
  # print(project_losses.values())
  # print(client_losses.values())
  plot_losses(project_losses, split='Distribution', mode='DWood', alias=alias)
  plot_client_losses(client_losses, 3, split='Distribution', mode='DWood', alias=alias, kcrossval=True)