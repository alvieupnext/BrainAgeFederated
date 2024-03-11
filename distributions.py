import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
#Obtain the distribution of the age of the patients
def age_distribution(data):
    return data['Age'].value_counts()


from typing import Tuple


def fit_gaussian_mixture(csv_path: str) -> Tuple[GaussianMixture, pd.DataFrame]:
    """
    This function takes the path to a CSV file, reads the 'Age' column,
    fits a Gaussian Mixture Model to the age data, and returns the fitted model
    along with the original data.

    Parameters:
    csv_path (str): The file path to the CSV file.

    Returns:
    Tuple[GaussianMixture, pd.DataFrame]: A tuple containing the fitted Gaussian Mixture Model
                                          and the DataFrame with the age data.
    """
    # Load the data
    data = pd.read_csv(csv_path)

    # Check if 'Age' column is in the data
    if 'Age' not in data.columns:
        raise ValueError("The CSV file does not contain an 'Age' column.")

    # Extract the 'Age' column
    ages = data['Age'].values.reshape(-1, 1)

    # Fit a Gaussian Mixture Model to the age data
    gmm = GaussianMixture(n_components=2)
    gmm.fit(ages)

    return gmm, data


#Call the functions
# plot_age_distribution(dataset)
# # plot_dataset_distribution(dataset)
# # plot_parent_dataset_distribution(dataset)
# print(age_distribution(dataset))
gmm, data = fit_gaussian_mixture('patients_dataset_9573.csv')

#Obtain the youngest and oldest ages
youngest_age = data['Age'].min()
oldest_age = data['Age'].max()
print(f"Youngest Age: {youngest_age}")
print(f"Oldest Age: {oldest_age}")

# Print the means and covariances of the Gaussian Mixture Model
print("Means:", gmm.means_)
print("Covariances:", gmm.covariances_)
print("Weights:", gmm.weights_)