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
            sample = self.sample(1)  # Assuming the distribution has a sample method that can sample individual values
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
        return samples

    def sample_with_limits(self, n, min_val, max_val):
        samples = []
        while len(samples) < n:
            # Generate one sample at a time for checking against limits
            sample, _ = self.distribution.sample(1)
            if min_val <= sample <= max_val:
                samples.append(sample)
        # Convert the list of arrays into a single 2D array
        return np.vstack(samples)

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

def gaussian_config(index, mean, cov):
    return f'Gaussian{index}', {'mean': mean, 'cov': cov}

def gaussian_mixture_config(n_components, random_state, df):
    data = df['Age'].values.reshape(-1, 1)
    return 'GaussianMixture', {'n_components': n_components, 'random_state': random_state, 'data': data}

df = pd.read_csv('patients_dataset_9573.csv')

normal_distribution1 = gaussian_config(1, 22.447, np.sqrt(8.41449796))
normal_distribution2 = gaussian_config(2, 22.447, np.sqrt(8.41449796))
normal_distribution3 = gaussian_config(3, 56.47287982, np.sqrt(260.14362206))
mixture_distribution = gaussian_mixture_config(2, 10, df)

#Function for plotting the age distribution
def plot_age_distribution(data, name):
    plt.figure(figsize=(10, 6))
    sns.histplot(data, kde=True, bins=30)
    plt.title(f'Age Distribution ({name})')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.show()

#Overall data
overall_data = []

for name, config in [normal_distribution1, normal_distribution2, normal_distribution3, mixture_distribution]:
    distribution = get_distribution(name, config)
    youngest_age = df['Age'].min()
    oldest_age = df['Age'].max()
    data = distribution.sample_with_limits(10000, youngest_age, oldest_age)
    plot_age_distribution(data, name)
    if name.startswith('Gaussian'):
        overall_data.append(data)

# print(overall_data)

#Flatten the numpy arrays into a single numpy array
overall_data = np.concatenate(overall_data)

#Plot the overall data
plot_age_distribution(overall_data, 'GaussianConcatenated')

plot_age_distribution(age_distribution(df), 'OriginalData')


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