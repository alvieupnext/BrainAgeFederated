import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture as GM

from plot import plot_dataset_distribution, plot_parent_dataset_distribution, plot_age_distribution


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
        # print(f"Mean: {self.mean}, Covariance: {self.cov}")

    def sample_with_limits(self, n, min_val, max_val):
        samples = []
        while len(samples) < n:
            sample = self.sample(1)[0]  # Assuming the distribution has a sample method that can sample individual values
            if min_val <= sample <= max_val:
                samples.append(sample)
        return np.array(samples)

    def sample(self, n):
        return np.random.normal(self.mean, self.cov, n)

class TwoGaussian(Gaussian):
    # Initialize the class with the mean, covariance, mean2, covariance2 and the probability to sample from the first distribution
    def __init__(self, mean, cov, mean2, cov2, p):
        super().__init__(mean, cov)
        self.mean2 = mean2
        self.cov2 = cov2
        self.p = p

    def sample1(self, n):
        return super().sample(n)

    # Sample from the distribution 2
    def sample2(self, n):
        return np.random.normal(self.mean2, self.cov2, n)

    def sample(self, n):
        # Sample from the first distribution with probability p
        if np.random.rand() < self.p:
            return self.sample1(n)
        # Sample from the second distribution with probability 1 - p
        return self.sample2(n)





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
    elif name.startswith('TwoGaussian'):
        mean = config['mean']
        cov = config['cov']
        mean2 = config['mean2']
        cov2 = config['cov2']
        p = config['p']
        return TwoGaussian(mean, cov, mean2, cov2, p)
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

def two_gaussian_config(mean, cov, mean2, cov2, p):
    percentage = int(p * 100)
    return f'TwoGaussian Junior {percentage}%, Senior {100 - percentage}%', {'mean': mean, 'cov': cov, 'mean2': mean2, 'cov2': cov2, 'p': p}

df = pd.read_csv('patients_dataset_9573.csv')

normal_distribution1 = gaussian_config(1, 22.447, np.sqrt(8.41449796))
normal_distribution2 = gaussian_config(2, 22.447, np.sqrt(8.41449796))
normal_distribution3 = gaussian_config(3, 56.47287982, np.sqrt(260.14362206))
mixture_distribution = gaussian_mixture_config(2, 10, df)
original_distribution = original_distribution_config(df)
# Make six nodes with the same distribution, transitioning from one to the other
two_gaussian_1 = two_gaussian_config(22.447, np.sqrt(8.41449796), 56.47287982, np.sqrt(260.14362206), 0.8)
two_gaussian_2 = two_gaussian_config(22.447, np.sqrt(8.41449796), 56.47287982, np.sqrt(260.14362206), 0.8)
two_gaussian_3 = two_gaussian_config(22.447, np.sqrt(8.41449796), 56.47287982, np.sqrt(260.14362206), 0.6)
two_gaussian_4 = two_gaussian_config(22.447, np.sqrt(8.41449796), 56.47287982, np.sqrt(260.14362206), 0.6)
two_gaussian_5 = two_gaussian_config(22.447, np.sqrt(8.41449796), 56.47287982, np.sqrt(260.14362206), 0.4)
two_gaussian_6 = two_gaussian_config(22.447, np.sqrt(8.41449796), 56.47287982, np.sqrt(260.14362206), 0.2)

#Given a dataframe and an age, retrieve all patients with that age
def retrieve_patients_with_closest_age(df, age, used_patients=None):
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

    # If there are used_patients, remove them from the DataFrame
    if used_patients is not None:
        df = df[~df['ID'].isin(used_patients)]

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
def dataset_from_distribution(df, distribution, n, resample=False, used_patients=None):
    #Create a new dataframe with the same columns as the original
    if used_patients is None:
        used_patients = set()
    new_df = pd.DataFrame(columns=df.columns)
    #Obtain the youngest and oldest ages
    youngest_age = df['Age'].min()
    oldest_age = df['Age'].max()
    # used_patients = set()
    for i in range(n):
        sample_patients = pd.DataFrame(columns=df.columns)
        while sample_patients.empty:
            #Sample the distribution with the limits
            sample_age = distribution.sample_with_limits(1, youngest_age, oldest_age)
            #Unnest the sample age
            sample_age = sample_age[0]
            if not resample:
                # From the sample age, retrieve all patients with that age
                sample_patients = retrieve_patients_with_closest_age(df, sample_age, used_patients=used_patients)
            else:
                #From the sample patients, remove all entries that have their ID in used_patients
                sample_patients = retrieve_patients_with_closest_age(df, sample_age, used_patients=None)
                # sample_patients = sample_patients[~sample_patients['ID'].isin(used_patients)]
        #Obtain a random patient from the sample patients
        patient = sample_patients.sample(n=1)
        #Add the patient to the new dataframe
        new_df = pd.concat([new_df, patient], ignore_index=True)
      #If resample is False, add the patient to used_patients
        used_patients.add(patient['ID'].values[0])
    return new_df, used_patients

#Returns n dataframes from the distribution, without resampling
#This function samples all patients at once per distribution
#Which gives the first distribution an unfair advantage
#However, the dataframes produced from this function are more likely to be statistically accurate
#As such, we use this function for the original, Gaussian and mixture distributions
#Distributions = name -> distribution object
def dataframes_from_distribution_full_sample(df, distributions):
    #For each distribution, create a dataset
    num_clients = len(distributions)
    patients_per_client = len(df) // num_clients
    result = {}
    used_patients = set()
    for name, distribution in distributions.items():
        dataset, used_patients = dataset_from_distribution(df, distribution, patients_per_client, resample=False,
                                                           used_patients=used_patients)
        result[name] = dataset
    return result


#Return n dataframes from the distribution, without resampling
#This function has the same functionality as dataset_from_distribution, but it samples one patient at a time
#This should give some distributions a fair chance to sample patients
#The previous function samples all patients at once per distribution, as it gives the first distribution an unfair advantage
#The disadvantage of this function is that it is slower, and it forces statistical unlikely samples to be taken
#due to how the current sampling is implemented
#As such this function is only used for the transition distributions, as they are a mixture of two distributions
#Distributions = name -> distribution object
def dataframes_from_distribution_fair_sample(df, distributions):
    num_clients = len(distributions)
    # patients_per_client = len(df) // num_clients
    #For each distribution, create a dataset
    datasets = [pd.DataFrame(columns=df.columns) for _ in range(num_clients)]
    # Obtain the youngest and oldest ages
    youngest_age = df['Age'].min()
    oldest_age = df['Age'].max()
    #Distribution names
    distribution_names = list(distributions.keys())
    #Distributions
    distribution_objects = list(distributions.values())
    #For every row in the dataframe
    used_patients = set()
    for i in range(len(df)):
        #Obtain the correct client index
        client_index = i % num_clients
        #Obtain the distribution object
        distribution = distribution_objects[client_index]
        #Sample the distribution with the limits
        sample_age = distribution.sample_with_limits(1, youngest_age, oldest_age)
        #Unnest the sample age
        sample_age = sample_age[0]
        #From the sample age, retrieve all patients with that age
        sample_patients = retrieve_patients_with_closest_age(df, sample_age, used_patients=used_patients)
        #Obtain a random patient from the sample patients
        patient = sample_patients.sample(n=1)
        #Add the patient to the new dataframe
        datasets[client_index] = pd.concat([datasets[client_index], patient], ignore_index=True)
        #Add the patient to used_patients
        used_patients.add(patient['ID'].values[0])
    result = {}
    for i in range(num_clients):
        result[distribution_names[i]] = datasets[i]
    return result

def dataframes_from_distribution(df, distribution_profile, nodes):
    distribution_profiles = distribution_profiles_6_nodes if nodes == 6 else distribution_profiles_3_nodes
    distributions = distribution_profiles[distribution_profile]
    if distribution_profile == 'Transition':
        return dataframes_from_distribution_fair_sample(df, distributions)
    return dataframes_from_distribution_full_sample(df, distributions)




#Function for plotting the age distribution
# def plot_age_distribution(data, name):
#     plt.figure(figsize=(10, 6))
#     sns.histplot(data, kde=True, bins=30)
#     plt.title(f'Age Distribution ({name})')
#     plt.xlabel('Age')
#     plt.ylabel('Frequency')
#     plt.show()

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

dataset = pd.read_csv('patients_dataset_9573.csv')

#Drop dataset and dataset_name columns
df = dataset.drop(columns=['dataset', 'dataset_name'])

#Split the dataframe into three dataframes, get the length of each dataframe
df_length = len(df) // 3

gaussian_young = get_distribution(*normal_distribution1)
gaussian_old = get_distribution(*normal_distribution3)
mixture = get_distribution(*mixture_distribution)
original = get_distribution(*original_distribution)

#Create a dictionary with the datasets
distribution_profiles_3_nodes = {'Original': {1: original, 2: original, 3: original},
                      'Gaussian': {'Junior1': gaussian_young, 'Junior2': gaussian_young, 'Senior': gaussian_old},
                      'Mixture': {1: mixture, 2: mixture, 3: mixture}
                                 }

distribution_profiles_6_nodes = {'Original': {1: original, 2: original, 3: original, 4: original, 5: original, 6: original},
                      'Gaussian': {'Junior1': gaussian_young, 'Junior2': gaussian_young,
                                   'Junior3': gaussian_young, 'Junior4': gaussian_young,
                                   'Senior1': gaussian_old,
                                   'Senior2': gaussian_old},
                      'Mixture': {1: mixture, 2: mixture, 3: mixture, 4: mixture, 5: mixture, 6: mixture},
                      'Transition': {'Junior/Senior (80%/20%) 1': get_distribution(*two_gaussian_1),
                                    'Junior/Senior (80%/20%) 2': get_distribution(*two_gaussian_2),
                                     'Junior/Senior (20%/80%)': get_distribution(*two_gaussian_6),
                                      'Junior/Senior (40%/60%)': get_distribution(*two_gaussian_5),
                                      'Junior/Senior (60%/40%) 1': get_distribution(*two_gaussian_3),
                                      'Junior/Senior (60%/40%) 2': get_distribution(*two_gaussian_4)
                                     },
                         }

if __name__ == '__main__':
    G1, _ = dataset_from_distribution(df, get_distribution(*normal_distribution1), int(len(df) * 0.625))
    # G2, _ = dataset_from_distribution(df, get_distribution(*normal_distribution2), df_length, resample=True)
    G3, _ = dataset_from_distribution(df, get_distribution(*normal_distribution3), int(len(df) * 0.375))
    G4, _ = dataset_from_distribution(df, get_distribution(*mixture_distribution), len(df))
    # # # #For all datasets, plot the age distribution
    plot_age_distribution(G1, 'utils/plots/age_distributionJunior.pdf', 'Junior')
    # plot_age_distribution(G2, 'utils/plots/age_distributionYoungNoResample2.pdf','Young2')
    plot_age_distribution(G3, 'utils/plots/age_distributionSenior.pdf', 'Senior')
    plot_age_distribution(G4, 'utils/plots/age_distributionMixture.pdf', 'Mixture')
    # mode = 'Gaussian'
    # #Obtain the original 6 nodes distribution
    # nodes = distribution_profiles_6_nodes[mode]
    # datasets = dataframes_by_distribution(df, nodes, mode)
    # for name, data in datasets.items():
    #     plot_age_distribution(data, None, name)
    # Create a general dictionary for the distributions

# mode = 'Gaussian'
# #Obtain the original 6 nodes distribution
# nodes = distribution_profiles_6_nodes[mode]
# datasets = dataframes_by_distribution(df, nodes, mode)
# for name, data in datasets.items():
#     plot_age_distribution(data, None, name)
#Create a general dictionary for the distributions\
# used_patients = set()

# used_patients = set()
# T1, used_patients = dataset_from_distribution(df, get_distribution(*two_gaussian_1), df_length, resample=False, used_patients=used_patients)
# T2, used_patients = dataset_from_distribution(df, get_distribution(*two_gaussian_2), df_length, resample=False, used_patients=used_patients)
# T6, used_patients = dataset_from_distribution(df, get_distribution(*two_gaussian_6), df_length, resample=False, used_patients=used_patients)
# T5, used_patients = dataset_from_distribution(df, get_distribution(*two_gaussian_5), df_length, resample=False, used_patients=used_patients)
# T3, used_patients = dataset_from_distribution(df, get_distribution(*two_gaussian_3), df_length, resample=False, used_patients=used_patients)
# T4, used_patients = dataset_from_distribution(df, get_distribution(*two_gaussian_4), df_length, resample=False, used_patients=used_patients)
#
# #For all datasets, plot the age distribution
# plot_age_distribution(T1, f'utils/plots/age_distribution{two_gaussian_1[0]}.pdf', two_gaussian_1[0])
# plot_age_distribution(T2, f'utils/plots/age_distribution{two_gaussian_2[0]}.pdf', two_gaussian_2[0])
# plot_age_distribution(T3, f'utils/plots/age_distribution{two_gaussian_3[0]}.pdf', two_gaussian_3[0])
# plot_age_distribution(T4, f'utils/plots/age_distribution{two_gaussian_4[0]}.pdf', two_gaussian_4[0])
# plot_age_distribution(T5, f'utils/plots/age_distribution{two_gaussian_5[0]}.pdf', two_gaussian_5[0])
# plot_age_distribution(T6, f'utils/plots/age_distribution{two_gaussian_6[0]}.pdf', two_gaussian_6[0])


# print(retrieve_patients_with_closest_age(df, 22.70).head())


#Call the functions
# plot_age_distribution(dataset, 'utils/plots/age_distribution.pdf')
# # plot_dataset_distribution(dataset, 'utils/plots/dataset_distribution.pdf')
# plot_parent_dataset_distribution(dataset, 'utils/plots/parent_dataset_distribution.pdf')
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