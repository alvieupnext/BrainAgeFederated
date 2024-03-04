# Based on the file content, we need to modify the function to correctly parse the losses.
# The losses are in the format "index,loss_value".
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from utils import generate_save_dir


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

strategies = ['FedAvg', 'FedProx']
model = ['DWood']
seeds = [2, 31, 35, 60, 67]
data = ['Dataset']

# Keep track of all the project names and the losses
project_losses = {}

# For every strategy, generate a model and for every DWood seed
for m in model:
  centralized_name = f'centralized_{m}'
  loss = get_centralized_losses(centralized_name)
  project_losses[centralized_name] = loss
  for s in strategies:
    for d in data:
      # Generate the project name
      project_name = f'{s}_{m}_{d}'
      #Make an array of the project names
      project_names = []
      # For every seed, add the seed to the project name if the mode is DWood
      if m == 'DWood':
        for seed in seeds:
          seed_name = f'_seed_{seed}'
          project_names.append(project_name + seed_name)
      else:
        project_names.append(project_name)
      # Get the losses for the project
      losses = [get_centralized_losses(p) for p in project_names]
      #Get the mean loss
      losses = np.array(losses).mean(axis=0)
      # Add the project name and the losses to the dictionary
      project_losses[project_name] = losses
      # for project_name in project_names:
      #   losses = get_centralized_losses_corrected(project_name)
      #   # Print the project name and the losses
      #   print(f'Project: {project_name}, Losses: {losses}')
      #   # Add the project name and the losses to the dictionary
      #   project_losses[project_name] = losses

# Define the plotting function
def plot_losses(projects_losses):
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

    # Customize the plot
    plt.title('Losses Over Server Rounds of Federated Learning', fontsize=16)
    plt.xlabel('Server Rounds', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(title='Project Names')
    sns.despine()

    # Show the plot
    plt.show()

# Plot the losses
plot_losses(project_losses)