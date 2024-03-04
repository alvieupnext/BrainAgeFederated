import os
import pandas as pd
def generate_save_dir(project_name):
  return default_location + project_name + "/"

project_name = 'FedProx_RW_Dataset_test'
default_location = './utils/models/'
save_dir = generate_save_dir(project_name)
#THIS IS THE RIGHT ONE
dwood = './utils/models/DWood/T1/'


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

def generate_project_name(strategy, mode, data_slice=None, seed=None):
  #Initial folder name
  project_name = f'{strategy}_{mode}'
  #If the strategy isn't centralized, add the data slice
  if strategy != 'centralized':
    #Data slice must be defined
    if data_slice is None:
      raise ValueError('Data slice must be defined for non-centralized strategies')
    project_name += f'_{data_slice}'
  #If the mode is DWood, add the seed
  if mode == 'DWood':
    #Seed must be defined
    if seed is None:
      raise ValueError('Seed must be defined for DWood mode')
    project_name += f'_seed_{seed}'
  return project_name
