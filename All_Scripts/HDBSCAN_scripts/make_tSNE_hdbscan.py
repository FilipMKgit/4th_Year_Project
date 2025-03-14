#this script applies HDBSCAN clustering to t-SNE-transformed data

from pathlib import Path #for handling file operations
import pandas as pd #for working with dataframes
import numpy as np #for numerical operations
import hdbscan #for clustering
from sklearn.preprocessing import StandardScaler #for standardizing data

################################################################
#input and output setup

tsne_dir = Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/tSNE_data") #directory with t-SNE-transformed data
input_dir = Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/working_preprocessed_data") #original preprocessed data dir
output_dir = Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/HDBSCAN_data/hdbscan_data_tSNE") #output folder
output_dir.mkdir(parents=True, exist_ok=True) #make sure output folder exists

#initialize HDBSCAN model
hdb = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5, metric='euclidean') #define clustering parameters

################################################################
#loop through t-SNE-transformed files

for tsne_file in tsne_dir.glob("*.csv"): #iterate over t-SNE-transformed data
    print(f"processing {tsne_file.name}...") #print current file being processed
    
    #match the original preprocessed file using its name
    preprocessed_name = tsne_file.stem.replace("_standardized_tSNE-transformed", "").replace("_tSNE-transformed", "") + ".csv" #extract original filename
    preprocessed_path = input_dir / preprocessed_name #full path to corresponding preprocessed file

    tsne_df = pd.read_csv(tsne_file) #load t-SNE-transformed data
    prepro_df = pd.read_csv(preprocessed_path) #load original preprocessed data

    #store original shape for debugging
    original_shape = prepro_df.shape  
    print(f"original shape: {original_shape}") #print initial dataframe shape
    
    #apply filtering based on FL2-H
    mask = prepro_df['FL2-H'] > 0.1 #create boolean mask for filtering
    filtered_tsne_df = tsne_df[mask] #apply filter to t-SNE data

    #store new shape after filtering
    new_shape = filtered_tsne_df.shape  
    print(f"after filtering: {new_shape}") #print new dataframe shape

    #skip files with insufficient data after filtering
    if len(filtered_tsne_df) < 5: #check if enough data is left
        print(f"skipping '{tsne_file.name}' due to insufficient data after filtering.") #print skip message
        continue #move to the next file

    ################################################################
    #scale the t-SNE data

    try:
        scaled_tsne_data = StandardScaler().fit_transform(filtered_tsne_df) #normalize data before clustering
    except ValueError as e:
        print(f"error scaling data for '{tsne_file.name}': {e}") #print error if scaling fails
        continue #skip to the next file

    ################################################################
    #apply HDBSCAN clustering

    try:
        clusters = hdb.fit_predict(scaled_tsne_data) #assign clusters
    except ValueError as e:
        print(f"error during hdbscan clustering for '{tsne_file.name}': {e}") #print error if clustering fails
        continue #skip to the next file

    #set all rows to -1 cluster initially
    prepro_df['Cluster'] = -1 #assign default value for all rows

    #update only filtered rows with their assigned cluster
    prepro_df.loc[filtered_tsne_df.index, 'Cluster'] = clusters #apply cluster labels

    #keep t-SNE columns in final output
    prepro_df[['tSNE-1', 'tSNE-2']] = tsne_df.iloc[:, :2] #retain t-SNE columns

    #define output file path
    output_filename = f"{preprocessed_path.stem}_hdbscan_tSNE.csv" #modify filename to indicate HDBSCAN clustering
    output_file_path = output_dir / output_filename #define full output path

    #save results
    prepro_df.to_csv(output_file_path, index=False) #save dataframe as csv file
    print(f"hdbscan clustering done for '{tsne_file.name}'. saved to: {output_file_path}") #print confirmation message

################################################################
#completion message

print("\033[1;32mall hdbscan clustering tasks finished! files are saved.\033[0m") #print success message
