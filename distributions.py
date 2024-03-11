import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture as GM
#Obtain the distribution of the age of the patients
def age_distribution(data):
    return data['Age'].values


from typing import Tuple

#Distribution Class for Gaussian
class Distribution:
    def sample(self, n):
        pass


#Gaussian class
class Gaussian(Distribution):
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov
        print(f"Mean: {self.mean}, Covariance: {self.cov}")

    def sample_with_limits(self, n, min_val, max_val):
        samples = []
        while len(samples) < n:
            sample = self.sample(1)[0]  # Assuming the distribution has a sample method that can sample individual values
            if min_val <= sample <= max_val:
                samples.append(sample)
        return np.array(samples)

    def sample(self, n):
        return np.random.normal(self.mean, self.cov, n)

#Gaussian Mixture class
class GaussianMixture(Distribution):
    def __init__(self, n_components, random_state):
        self.distribution = GM(n_components=n_components)

    def fit(self, X):
        self.distribution.fit(X)

    def sample(self, n):
        samples, _ = self.distribution.sample(n)
        #Unflatten the samples
        return [sample[0] for sample in samples]

    def sample_with_limits(self, n, min_val, max_val):
        samples = []
        while len(samples) < n:
            # Generate one sample at a time for checking against limits
            sample = self.sample(1)
            sample = sample[0]
            if min_val <= sample <= max_val:
                samples.append(sample)
        # Convert the list of arrays into a single 2D array
        return samples

class OriginalDistribution(Distribution):
    def __init__(self, df):
        self.df = df

    def sample(self, n):
        return self.df.sample(n)['Age'].values

    def sample_with_limits(self, n, min_val, max_val):
        return self.df[(self.df['Age'] >= min_val) & (self.df['Age'] <= max_val)].sample(n)['Age'].values

#Flatten a single element nested in one of multiple nestings
# def flatten_single_element(nested):
#     if isinstance(nested, (list, tuple)):
#         if len(nested) == 1:
#             return flatten_single_element(nested[0])
#         else:
#             return type(nested)(flatten_single_element(n) for n in nested)
#     else:
#         return nested

#Obtain distribution from name
def get_distribution(name, config):
    if name == 'GaussianMixture':
        #Obtain the number of components and the random state from the config
        n_components = config['n_components']
        random_state = config['random_state']
        #Create the Gaussian Mixture distribution
        gm = GaussianMixture(n_components, random_state)
        #Retreive the data from the config
        X = config['data']
        #Fit the Gaussian Mixture to the data
        gm.fit(X)
        return gm
    #If the name begins with Gaussian
    elif name.startswith('Gaussian'):
        #Obtain the mean and covariance from the config
        mean = config['mean']
        cov = config['cov']
        #Create the Gaussian distribution
        return Gaussian(mean, cov)
    else:
        df = config['df']
        return OriginalDistribution(df)

def gaussian_config(index, mean, cov):
    return f'Gaussian_{mean}_{index}', {'mean': mean, 'cov': cov}

def gaussian_mixture_config(n_components, random_state, df):
    data = df['Age'].values.reshape(-1, 1)
    return 'GaussianMixture', {'n_components': n_components, 'random_state': random_state, 'data': data}

def original_distribution_config(df):
    return 'OriginalDistribution', {'df': df}

df = pd.read_csv('patients_dataset_9573.csv')

normal_distribution1 = gaussian_config(1, 22.447, np.sqrt(8.41449796))
normal_distribution2 = gaussian_config(2, 22.447, np.sqrt(8.41449796))
normal_distribution3 = gaussian_config(3, 56.47287982, np.sqrt(260.14362206))
mixture_distribution = gaussian_mixture_config(2, 10, df)
original_distribution = original_distribution_config(df)

#Given a dataframe and an age, retrieve all patients with that age
def retrieve_patients_with_closest_age(df, age):
    """
    Retrieve patients with a given age or the closest age from a DataFrame.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing patient data.
    - age (int): The target age to search for.

    Returns:
    - pandas.DataFrame: Patients with the exact or closest age.
    """
    # Check if the DataFrame is empty
    if df.empty:
        return df  # Return the empty DataFrame directly

    # Find patients with the exact age
    patients = df.loc[df['Age'] == age]

    # If no patients were found with that exact age
    if patients.empty:
        # Calculate the absolute difference only once
        abs_diff = (df['Age'] - age).abs()
        closest_age = df.loc[abs_diff.idxmin(), 'Age']

        # Retrieve patients with the closest age
        patients = df.loc[df['Age'] == closest_age]

    return patients


# Given a dataset, distribution and n samples, create a new dataset with the same distribution
def dataset_from_distribution(df, distribution, n, resample=True):
    #Create a new dataframe with the same columns as the original
    new_df = pd.DataFrame(columns=df.columns)
    #Obtain the youngest and oldest ages
    youngest_age = df['Age'].min()
    oldest_age = df['Age'].max()
    used_patients = set()
    for i in range(n):
        sample_patients = pd.DataFrame(columns=df.columns)
        while sample_patients.empty:
            #Sample the distribution with the limits
            sample_age = distribution.sample_with_limits(1, youngest_age, oldest_age)
            #Unnest the sample age
            sample_age = sample_age[0]
            #From the sample age, retrieve all patients with that age
            sample_patients = retrieve_patients_with_closest_age(df, sample_age)
            if not resample:
                #From the sample patients, remove all entries that have their ID in used_patients
                sample_patients = sample_patients[~sample_patients['ID'].isin(used_patients)]
        #Obtain a random patient from the sample patients
        patient = sample_patients.sample(n=1)
        #Add the patient to the new dataframe
        new_df = pd.concat([new_df, patient], ignore_index=True)
      #If resample is False, add the patient to used_patients
        if not resample:
            used_patients.add(patient['ID'].values[0])
    return new_df



#Function for plotting the age distribution
def plot_age_distribution(data, name):
    plt.figure(figsize=(10, 6))
    sns.histplot(data, kde=True, bins=30)
    plt.title(f'Age Distribution ({name})')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.show()

#Overall data
# overall_data = []

# for name, config in [normal_distribution1, normal_distribution2, normal_distribution3, mixture_distribution]:
#     distribution = get_distribution(name, config)
#     youngest_age = df['Age'].min()
#     oldest_age = df['Age'].max()
#     data = distribution.sample_with_limits(10000, youngest_age, oldest_age)
#     plot_age_distribution(data, name)
#     if name.startswith('Gaussian'):
#         overall_data.append(data)

# print(overall_data)

#Flatten the numpy arrays into a single numpy array
# overall_data = np.concatenate(overall_data)

#Plot the overall data
# plot_age_distribution(overall_data, 'GaussianConcatenated')
#
# plot_age_distribution(age_distribution(df), 'OriginalData')

df = pd.read_csv('patients_dataset_9573.csv')

#Drop dataset and dataset_name columns
df = df.drop(columns=['dataset', 'dataset_name'])

#Split the dataframe into three dataframes, get the length of each dataframe
df_length = len(df) // 3

gaussian_young = get_distribution(*normal_distribution1)
gaussian_old = get_distribution(*normal_distribution3)
mixture = get_distribution(*mixture_distribution)
original = get_distribution(*original_distribution)

#Create a dictionary with the datasets
distribution_profiles = {'Original': {1: original, 2: original, 3: original},
                      'Gaussian': {'Young1': gaussian_young, 'Young2': gaussian_young, 'Old': gaussian_old},
                      'Mixture': {1: mixture, 2: mixture, 3: mixture}}

#Create a general dictionary for the distributions

# G1 = dataset_from_distribution(df, get_distribution(*normal_distribution1), df_length)
# G2 = dataset_from_distribution(df, get_distribution(*normal_distribution2), df_length)
# G3 = dataset_from_distribution(df, get_distribution(*normal_distribution3), df_length)
# G4 = dataset_from_distribution(df, get_distribution(*mixture_distribution), df_length)
#
# #For all datasets, plot the age distribution
# plot_age_distribution(age_distribution(G1), 'G1')
# plot_age_distribution(age_distribution(G2), 'G2')
# plot_age_distribution(age_distribution(G3), 'G3')
# plot_age_distribution(age_distribution(G4), 'G4')


# print(retrieve_patients_with_closest_age(df, 22.70).head())


#Call the functions
# plot_age_distribution(dataset)
# # plot_dataset_distribution(dataset)
# # plot_parent_dataset_distribution(dataset)
# print(age_distribution(dataset))
# gmm, data = fit_gaussian_mixture('patients_dataset_9573.csv')
#
# #Obtain the youngest and oldest ages
# youngest_age = data['Age'].min()
# oldest_age = data['Age'].max()
# print(f"Youngest Age: {youngest_age}")
# print(f"Oldest Age: {oldest_age}")
#
# # Print the means and covariances of the Gaussian Mixture Model
# print("Means:", gmm.means_)
# print("Covariances:", gmm.covariances_)
# print("Weights:", gmm.weights_)
#
# gmm.sample