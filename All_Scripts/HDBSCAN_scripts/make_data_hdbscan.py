#this script applies HDBSCAN clustering to preprocessed data

from pathlib import Path #for handling file operations
import pandas as pd #for working with dataframes
import numpy as np #for numerical operations
from hdbscan import HDBSCAN #for HDBSCAN clustering

################################################################
#input and output setup

input_dir = Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/working_preprocessed_data") #directory with input csv files
output_dir = Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/HDBSCAN_data/hdbscan_data") #output folder
output_dir.mkdir(parents=True, exist_ok=True) #make sure output folder exists

################################################################
#loop through each CSV file in the input directory

for file_path in input_dir.glob("*.csv"): #iterate over csv files
    print(f"processing {file_path.name}...") #print current file being processed
    
    df = pd.read_csv(file_path) #load csv file into dataframe
    df = df.copy() #copy dataframe to avoid modifying the original file

    #store original shape for debugging
    original_shape = df.shape  
    print(f"original shape: {original_shape}") #print initial dataframe shape
    
    #filter out rows where FL2-H is too low
    filtered_df = df[df['FL2-H'] > 0.1] #only keep rows where FL2-H > 0.1
    
    #store new shape after filtering
    new_shape = filtered_df.shape  
    print(f"after filtering: {new_shape}") #print new dataframe shape

    #skip files with insufficient data after filtering
    if len(filtered_df) < 5: #check if enough data is left
        print(f"skipping '{file_path.name}' due to insufficient data after filtering.") #print skip message
        continue #move to the next file

    ################################################################
    #apply HDBSCAN clustering

    hdbscan = HDBSCAN(min_cluster_size=100, min_samples=5) #initialize HDBSCAN model
    clusters = hdbscan.fit_predict(filtered_df) #assign clusters to data points

    #set all rows to -1 cluster initially
    df['Cluster'] = -1 #assign default value for all rows

    #update only filtered rows with their assigned cluster
    df.loc[filtered_df.index, 'Cluster'] = clusters #apply cluster labels

    #define output file path
    output_filename = f"{file_path.stem}_hdbscan.csv" #modify filename to indicate HDBSCAN clustering
    output_file_path = output_dir / output_filename #define full output path

    #save results
    df.to_csv(output_file_path, index=False) #save dataframe as csv file
    print(f"hdbscan clustering done for '{file_path.name}'. saved to: {output_file_path}") #print confirmation message

################################################################
#completion message

print("\033[1;32mall hdbscan clustering tasks finished! files are saved.\033[0m") #print success message
