#this script applies Gaussian Mixture Model (GMM) clustering to preprocessed data

from pathlib import Path #for handling file and folder operations
import pandas as pd #for working with dataframes
from sklearn.mixture import GaussianMixture #for performing Gaussian Mixture Model clustering
from sklearn.preprocessing import StandardScaler #for standardizing data
import numpy as np #for numerical operations

################################################################
# Define the input and output directories

preprocessed_data_dir = Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/working_preprocessed_data") #directory for preprocessed data
gmm_output_dir = Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/GMM_data/gmm_data") #output directory for GMM results

################################################################
# Ensure the output directory exists

gmm_output_dir.mkdir(parents=True, exist_ok=True) #create directory if it doesn't exist

################################################################
# Loop through all preprocessed files and perform GMM clustering

for file_path in preprocessed_data_dir.glob("*.csv"): #iterate over all CSV files
    print(f"processing {file_path.name}...") #print current file being processed

    ################################################################
    # Load data

    preprocessed_data = pd.read_csv(file_path) #load CSV file into a dataframe
    preprocessed_data = preprocessed_data.copy() #copy dataframe to avoid modifying the original file

    #store original shape for debugging
    original_shape = preprocessed_data.shape  
    print(f"original shape: {original_shape}") #print initial dataframe shape

    ################################################################
    # Filter rows where FL2-H > 0.1

    mask = preprocessed_data['FL2-H'] > 0.1 #create boolean mask for filtering
    filtered_data = preprocessed_data[mask][['FL3-H', 'FL2-H']] #keep only selected features

    #store new shape after filtering
    new_shape = filtered_data.shape  
    print(f"after filtering: {new_shape}") #print new dataframe shape

    ################################################################
    # Check if there is enough data to cluster

    if len(filtered_data) < 5: #ensure enough data exists for clustering
        print(f"skipping '{file_path.name}' due to insufficient data after filtering.") #print skip message
        continue #move to the next file

    ################################################################
    # Apply log transformation and scale the data

    try:
        filtered_data = np.log10(filtered_data + 1) #apply log transformation to stabilize variance
        scaled_data = StandardScaler().fit_transform(filtered_data) #normalize data before clustering
    except ValueError as e:
        print(f"error applying log transformation for '{file_path.name}': {e}") #print error message
        continue #skip to the next file

    ################################################################
    # Perform Gaussian Mixture Model clustering with regularization

    gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=100) #initialize GMM model
    try:
        clusters = gmm.fit_predict(scaled_data) #assign clusters
    except ValueError as e:
        print(f"error during GMM clustering for '{file_path.name}': {e}") #print error message if clustering fails
        continue #skip to the next file

    ################################################################
    # Add cluster assignments

    preprocessed_data['Cluster'] = -1 #assign default cluster value (-1) to all rows
    preprocessed_data.loc[mask, 'Cluster'] = clusters #apply cluster labels only to filtered rows

    ################################################################
    # Save the clustered data

    output_file_name = f"{file_path.stem}_gmm_data.csv" #construct filename for output
    output_file_path = gmm_output_dir / output_file_name #define full output path

    preprocessed_data.to_csv(output_file_path, index=False) #save dataframe as CSV file
    print(f"GMM clustering completed for '{file_path.name}'. Results saved to: {output_file_path}") #print confirmation message

################################################################
# Print a success message

print("\033[1;32m=======================\n      GMM Completed!       \n=======================\033[0m") #print success message
