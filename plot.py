# Based on the file content, we need to modify the function to correctly parse the losses.
# The losses are in the format "index,loss_value".
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from utils import generate_save_dir, plot_folder


def get_centralized_losses(project_name):
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

strategies = ['FedAvg', 'FedProx']
model = ['DWood']
seeds = [2]
# data = ['Dataset']
data = ['Distribution_Gaussian', 'Distribution_Original']
# Keep track of all the project names and the losses
project_losses = {}
client_losses = {}

# For every strategy, generate a model and for every DWood seed
for m in model:
  centralized_name = f'centralized_{m}'
  loss = get_centralized_losses(centralized_name)
  project_losses[centralized_name] = loss
  for s in strategies:
    for d in data:
      strategy_data_name = f'{s}_{m}'
      # Generate the project name
      project_name = f'{strategy_data_name}_{d}'
      #Make an array of the project names
      project_names = []
      # For every seed, add the seed to the project name if the mode is DWood
      if m == 'DWood':
        for seed in seeds:
          seed_name = f'_seed_{seed}'
          project_names.append(project_name + seed_name + '_ascending_epochs')
      else:
        project_names.append(project_name)
      # Get the losses for the project
      losses = [get_centralized_losses(p) for p in project_names]
      #Get the mean loss
      losses = np.array(losses).mean(axis=0)
      # Add the project name and the losses to the dictionary
      project_losses[project_name] = losses
      # Get the clients, training, validation losses and x-ticks from the project
      client_tuples = [get_decentralized_losses(p) for p in project_names]
      # Generate 4 different arrays for the clients, training, validation losses and x-ticks
      clients, training_losses, validation_losses, x_ticks = zip(*client_tuples)
      #For the clients and x-ticks, take the first array in the list and use that
      clients = clients[0]
      x_ticks = x_ticks[0]
      #Take the mean of the training and validation losses
      training_losses = np.array(training_losses).mean(axis=0)
      validation_losses = np.array(validation_losses).mean(axis=0)
      #Add these 4 things to the client_losses dictionary
      client_losses[strategy_data_name] = (clients, training_losses, validation_losses, x_ticks)
      # for project_name in project_names:
      #   losses = get_centralized_losses_corrected(project_name)
      #   # Print the project name and the losses
      #   print(f'Project: {project_name}, Losses: {losses}')
      #   # Add the project name and the losses to the dictionary
      #   project_losses[project_name] = losses

# Define the plotting function
def plot_losses(projects_losses, split='Dataset', mode='DWood'):
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
    plt.title(f'Losses Over Server Rounds of Federated Learning ({split}, {verbose_mode})', fontsize=16)
    plt.xlabel('Server Rounds', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(title='Strategies')
    sns.despine()
    #Save the plot at the plot folders
    path = os.path.join(plot_folder, f'losses_over_server_rounds_{split}_{mode}.pdf')
    #Save as plot
    # plt.savefig(path, format='pdf', dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()


def plot_client_losses(client_losses, client_number=12, split='Dataset', mode='DWood'):
  # Set the visual theme of the plots with Seaborn
  sns.set(style="whitegrid")

  #Get the amouunt of clients and turn it into a rows and column
  #The amount of rows varies and the columns are always 4
  rows = int(np.ceil(client_number / 4))
  columns = 4


  # Create a 4x3 grid of subplots with shared x and y axes for uniform scale
  fig, axes = plt.subplots(rows, columns, figsize=(14,14), sharex=True, sharey=True)
  axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration

  # Iterate through each project in the client_losses dictionary
  for project_name, (clients, training_losses, validation_losses, x_ticks) in client_losses.items():
    # Iterate through each client and corresponding subplot axis
    for idx, (client, ax) in enumerate(zip(clients, axes)):
      # Plot the training loss for the current client
      sns.lineplot(x=x_ticks[idx], y=training_losses[idx], ax=ax, label=f'{project_name} Train Loss', marker='o')
      # Plot the validation loss for the current client with a dotted line
      sns.lineplot(x=x_ticks[idx], y=validation_losses[idx], ax=ax, label=f'{project_name} Val Loss', marker='s',
                   linestyle=':')
      # Set the title of the subplot to the client's name
      ax.set_title(f'{client}')
      # Set the labels for the x and y axes
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
  plt.suptitle(f'Losses by Client ({split}, {verbose_mode})', fontsize=20)
  # Adjust the layout to make sure there's no overlap between subplots
  plt.tight_layout()
  path = os.path.join(plot_folder, f'Losses_by_client_{split}_{mode}.pdf')
  #Save the plot in the utils folder
  # plt.savefig(path, format='pdf', dpi=300, bbox_inches='tight')
  # Display the plot
  plt.show()

# #If the plot folder does not exist, create it
# if not os.path.exists(plot_folder):
#   os.makedirs(plot_folder)
# # Plot the losses
# plot_losses(project_losses, split='Distribution', mode='DWood')
# plot_client_losses(client_losses,4, split='Distribution', mode='DWood')
#
# # Read patients_dataset_9573_noage.csv
# dataset = pd.read_csv('patients_dataset_9573.csv')

# Plot the Age distribution, make it a function
# Add spacing between the bins
def plot_age_distribution(data, save_path=None):
  plt.figure(figsize=(10, 6))  # Increase figure size
  sns.set(style="whitegrid")  # Use seaborn style for prettier plots
  data['Age'].plot(kind='hist', bins=20, title='Age Distribution', color='skyblue', edgecolor='black', rwidth=0.8)
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
def plot_dataset_distribution(data):
    value_counts = data['dataset_name'].value_counts()
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
    plt.show()


def plot_parent_dataset_distribution(data):
    plt.figure(figsize=(10, 6))  # Increase figure size
    sns.set(style="whitegrid")  # Use seaborn style
    data['dataset'].value_counts().plot(kind='bar', title='Parent Dataset Distribution', color='lightcoral', edgecolor='black')
    plt.ylabel('Amount of Patients')
    plt.xlabel('Dataset')
    plt.xticks(rotation=45, ha='right')  # Rotate and align labels
    plt.grid(axis='y')  # Add horizontal grid lines
    plt.tight_layout()  # Adjust layout
    plt.show()
