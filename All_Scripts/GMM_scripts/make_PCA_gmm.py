#this script applies Gaussian Mixture Model (GMM) clustering to PCA-transformed data

from pathlib import Path #for handling file operations
import pandas as pd #for working with dataframes
import numpy as np #for numerical operations
from sklearn.mixture import GaussianMixture #for GMM clustering
from sklearn.preprocessing import StandardScaler #for data scaling

################################################################
#input and output setup

pca_dir = Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/PCA_data") #directory with PCA-transformed data
input_dir = Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/working_preprocessed_data") #original preprocessed data dir
output_dir = Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/GMM_data/gmm_data_PCA") #output folder
output_dir.mkdir(parents=True, exist_ok=True) #make sure output folder exists

################################################################
#loop through PCA-transformed files

for pca_file in pca_dir.glob("*.csv"): #iterate over PCA-transformed data
    print(f"processing {pca_file.name}...") #print current file being processed
    
    #match the original preprocessed file using its name
    preprocessed_name = pca_file.stem.replace("_standardized_PCA-transformed", "") + ".csv" #extract original filename
    preprocessed_path = input_dir / preprocessed_name #full path to corresponding preprocessed file

    pca_df = pd.read_csv(pca_file) #load PCA-transformed data
    prepro_df = pd.read_csv(preprocessed_path) #load original preprocessed data

    #store original shape for debugging
    original_shape = prepro_df.shape  
    print(f"original shape: {original_shape}") #print initial dataframe shape
    
    #apply filtering based on FL2-H
    mask = prepro_df['FL2-H'] > 0.1 #create boolean mask for filtering
    filtered_pca_df = pca_df[mask] #apply filter to PCA data

    #store new shape after filtering
    new_shape = filtered_pca_df.shape  
    print(f"after filtering: {new_shape}") #print new dataframe shape

    #skip files with insufficient data after filtering
    if len(filtered_pca_df) < 5: #check if enough data is left
        print(f"skipping '{pca_file.name}' due to insufficient data after filtering.") #print skip message
        continue #move to the next file

    ################################################################
    #apply standard scaling

    try:
        scaled_pca_data = StandardScaler().fit_transform(filtered_pca_df) #normalize data before clustering
    except ValueError as e:
        print(f"error scaling data for '{pca_file.name}': {e}") #print error message
        continue #skip to the next file

    ################################################################
    #apply GMM clustering

    gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=100) #initialize GMM model
    try:
        clusters = gmm.fit_predict(scaled_pca_data) #assign clusters
    except ValueError as e:
        print(f"error during gmm clustering for '{pca_file.name}': {e}") #print error message if clustering fails
        continue #skip to the next file

    #set all rows to -1 cluster initially
    prepro_df['Cluster'] = -1 #assign default value for all rows

    #update only filtered rows with their assigned cluster
    prepro_df.loc[filtered_pca_df.index, 'Cluster'] = clusters #apply cluster labels

    #keep PCA columns in final output
    pca_columns = [col for col in pca_df.columns if "PC" in col] #find PCA columns
    prepro_df[pca_columns] = pca_df[pca_columns] #add them back to the dataframe

    #define output file path
    output_filename = f"{preprocessed_path.stem}_gmm_PCA.csv" #modify filename to indicate GMM clustering
    output_file_path = output_dir / output_filename #define full output path

    #save results
    prepro_df.to_csv(output_file_path, index=False) #save dataframe as csv file
    print(f"gmm clustering done for '{pca_file.name}'. saved to: {output_file_path}") #print confirmation message

################################################################
#completion message

print("\033[1;32mall gmm clustering tasks finished! files are saved.\033[0m") #print success message
