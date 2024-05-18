import os
import pandas as pd
def generate_save_dir(project_name):
  return default_location + project_name + "/"

project_name = 'FedProx_RW_Dataset_test'
default_location = './utils/models/'
save_dir = generate_save_dir(project_name)
#THIS IS THE RIGHT ONE
dwood = './utils/models/DWood/T1/'
andrei = './utils/models/AndSilva/'

#Plot folder
plot_folder = './utils/plots/'
test_folder = './utils/tests/'

training_dataset = 'patients_dataset_9573_train.csv'
testing_dataset = 'patients_dataset_9573_test.csv'


def save_csv_prediction(outdir, project_name, true_ages, pred_ages, ids_sub):
  outname = project_name + "_brain_age_output.csv"
  fullname = outdir + outname
  pd.DataFrame(
    {'ID': ids_sub,
     'Chronological age': true_ages,
     'Predicted_age (years)': pred_ages}
  ).set_index('ID').to_csv(fullname)
  print('csv file with predictions saved in ' + str(fullname))

  #Function for retreiving the path to a pt file in a folder
def get_pt_file_path(project_name, model_path=default_location):
  path = os.path.join(model_path, project_name)
  #List all the pt files present in the folder
  pt_files = [f for f in os.listdir(path) if f.endswith('.pt')]
  #Sort them by date
  pt_files.sort(key=lambda x: os.path.getmtime(path + '/' + x))
  #Get the path of the latest one
  return path + '/' + pt_files[-1]

def load_andrei_model_paths(prefix='rw', model_path=andrei):
  #From the model path, get all the folders that start with the prefix
  folders = [f for f in os.listdir(model_path) if f.startswith(prefix)]
  #From each folder, get the path to the latest pt file
  return [get_pt_file_path(f, model_path) for f in folders]


def generate_project_name(strategy, mode, data_slice=None, seed=None, distribution=None, alias=None):
  #Initial folder name
  project_name = f'{strategy}_{mode}'
  #If the strategy isn't centralized, add the data slice
  if strategy != 'centralized':
    #Data slice must be defined
    if data_slice is None:
      raise ValueError('Data slice must be defined for non-centralized strategies')
    if distribution is not None:
      project_name += f'_{data_slice}_{distribution}'
    else:
      project_name += f'_{data_slice}'
  #If the mode is DWood, add the seed
  if mode == 'DWood':
    #Seed must be defined
    if seed is None:
      raise ValueError('Seed must be defined for DWood mode')
    project_name += f'_seed_{seed}'
  #If an alias is defined, add it
  if alias is not None:
    project_name += f'_{alias}'
  return project_name

#From a project name, return the strategy, mode, data slice, seed, distribution and alias
def parse_project_name(project_name):
  #Split the project name by _
  parts = project_name.split('_')
  # print(parts)
  #Get the strategy
  strategy = parts[0]
  #Get the mode
  mode = parts[1]
  #If the strategy isn't centralized, get the data slice
  if strategy != 'centralized':
    data_slice = parts[2]
    #If the data slice is distribution, get the distribution
    if data_slice == 'Distribution':
      distribution = parts[3]
    else:
      distribution = None
  else:
    data_slice = None
    distribution = None
  #If the mode is DWood, get the seed
  if mode == 'DWood':
    seed = int(parts[5])
  else:
    seed = None
  #If there is an alias, get it
  if len(parts) > 3 and parts[-1] != 'seed':
    node = int(parts[-2])
  else:
    node = None
  return strategy, mode, data_slice, seed, distribution, node

