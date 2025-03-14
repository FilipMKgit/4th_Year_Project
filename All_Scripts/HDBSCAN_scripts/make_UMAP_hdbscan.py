#this script applies HDBSCAN clustering to UMAP-transformed data
from pathlib import Path #for handling file operations
import pandas as pd #for working with dataframes
from hdbscan import HDBSCAN #for clustering

################################################################
#input and output setup

umap_dir = Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/UMAP_data") #directory with UMAP-transformed data
input_dir = Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/working_preprocessed_data") #original preprocessed data dir
output_dir = Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/HDBSCAN_data/hdbscan_data_UMAP") #output folder
output_dir.mkdir(parents=True, exist_ok=True) #make sure output folder exists

################################################################
#loop through UMAP-transformed files

for umap_file in umap_dir.glob("*.csv"): #iterate over UMAP-transformed data
    print(f"processing {umap_file.name}...") #print current file being processed
    
    #match the original preprocessed file using its name
    preprocessed_name = umap_file.stem.replace("_standardized_UMAP-transformed", "").replace("_UMAP-transformed", "") + ".csv" #extract original filename
    preprocessed_path = input_dir / preprocessed_name #full path to corresponding preprocessed file

    umap_df = pd.read_csv(umap_file) #load UMAP-transformed data
    prepro_df = pd.read_csv(preprocessed_path) #load original preprocessed data

    #store original shape for debugging
    original_shape = prepro_df.shape  
    print(f"original shape: {original_shape}") #print initial dataframe shape
    
    #apply filtering based on FL2-H
    mask = prepro_df['FL2-H'] > 0.1 #create boolean mask for filtering
    filtered_umap_df = umap_df[mask] #apply filter to UMAP data

    #store new shape after filtering
    new_shape = filtered_umap_df.shape  
    print(f"after filtering: {new_shape}") #print new dataframe shape

    #skip files with insufficient data after filtering
    if len(filtered_umap_df) < 5: #check if enough data is left
        print(f"skipping '{umap_file.name}' due to insufficient data after filtering.") #print skip message
        continue #move to the next file

    ################################################################
    #apply HDBSCAN clustering

    try:
        hdbscan = HDBSCAN(min_cluster_size=29, min_samples=5) #initialize HDBSCAN model
        clusters = hdbscan.fit_predict(filtered_umap_df) #assign clusters
    except ValueError as e:
        print(f"error during hdbscan clustering for '{umap_file.name}': {e}") #print error message if clustering fails
        continue #skip to the next file

    #set all rows to -1 cluster initially
    prepro_df['Cluster'] = -1 #assign default value for all rows

    #update only filtered rows with their assigned cluster
    prepro_df.loc[filtered_umap_df.index, 'Cluster'] = clusters #apply cluster labels

    #keep UMAP columns in final output
    prepro_df[['UMAP-1', 'UMAP-2']] = umap_df.iloc[:, :2] #retain UMAP columns

    #define output file path
    output_filename = f"{preprocessed_path.stem}_hdbscan_UMAP.csv" #modify filename to indicate HDBSCAN clustering
    output_file_path = output_dir / output_filename #define full output path

    #save results
    prepro_df.to_csv(output_file_path, index=False) #save dataframe as csv file
    print(f"hdbscan clustering done for '{umap_file.name}'. saved to: {output_file_path}") #print confirmation message

################################################################
#completion message

print("\033[1;32mall hdbscan clustering tasks finished! files are saved.\033[0m") #print success message
