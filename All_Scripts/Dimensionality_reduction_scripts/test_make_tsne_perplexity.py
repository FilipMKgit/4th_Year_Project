#this script tests different perplexity values for t-SNE on a random subset of standardized data
#it selects 5 random files and applies t-SNE using multiple perplexity values

from pathlib import Path #for handling file operations
import pandas as pd #for working with dataframes
import numpy as np #for numerical operations
import random #for selecting random files
from sklearn.manifold import TSNE #for t-SNE transformation

################################################################
# define input and output directories

input_dir = Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/working_data_standardized") #where standardized data is stored
output_base_dir = Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/tSNE_test_data") #where test outputs will be saved
output_base_dir.mkdir(parents=True, exist_ok=True)  #make sure output directory exists

# get a list of all CSV files in the input directory
csv_files = list(input_dir.glob("*.csv")) #find all CSV files

# randomly select 5 files for testing
random.seed(42)  #ensures results are reproducible
selected_files = random.sample(csv_files, 5) #randomly pick 5 files

################################################################
# define parameters for t-SNE transformation

perplexities = [2, 5, 30, 50, 100]  #different perplexity values to test

#columns to use for t-SNE
feature_cols = [ "FSC-A", "FSC-H", "SSC-A", "SSC-H","FL1-A", "FL1-H", "FL2-A", "FL2-H", "FL3-A", "FL3-H", "FL4-A", "FL4-H"]
#list of features used for t-SNE

print("executing t-SNE transformation...") #exectution message

################################################################
# process each selected CSV file

for file_path in selected_files: #loop through randomly selected files
    print(f"processing {file_path.name}...") #print current file being processed
    
    df = pd.read_csv(file_path) #load the CSV file into a dataframe
    df = df.copy() #make a copy to avoid modifying the original data
    
    #store original shape for debugging
    original_shape = df.shape  
    print(f"original shape: {original_shape}") #print dataframe shape
    
    #check if 'Cluster' and 'Label' columns exist, store them if present
    cluster_label_cols = [col for col in ['Cluster', 'Label'] if col in df.columns] #identify label columns if present
    
    #filter only the columns needed for t-SNE
    tsne_input_cols = [col for col in feature_cols if col in df.columns] #ensure only valid feature columns are used
    df_tsne_input = df[tsne_input_cols].dropna() #subset dataframe and drop missing values
    
    #store new shape after dropping NaNs
    new_shape = df_tsne_input.shape  
    print(f"after NaN removal: {new_shape}") #print new dataframe shape

    ################################################################
    # apply t-SNE for each perplexity value
    
    for perplexity in perplexities: #loop through different perplexity values
        output_dir = output_base_dir / f"tSNE_test_p={perplexity}" #define output directory for current perplexity
        output_dir.mkdir(parents=True, exist_ok=True)  #make sure directory exists

        #initialize and apply t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=400, random_state=42) #set t-SNE parameters
        transformed_data = tsne.fit_transform(df_tsne_input) #apply t-SNE transformation

        #convert result to a dataframe
        tsne_df = pd.DataFrame(transformed_data, columns=['tSNE1', 'tSNE2']) #store results in dataframe

        #if 'Cluster' and 'Label' exist, add them back to the output
        if cluster_label_cols: #check if original file had labels
            tsne_df = pd.concat([df[cluster_label_cols].reset_index(drop=True), tsne_df], axis=1) #merge labels back in

        #construct output filename
        output_file_name = file_path.stem.replace("_standardized", "") + f"_tSNE-transformed_P={perplexity}.csv" #modify filename
        output_file_path = output_dir / output_file_name #define full path

        #save the transformed data
        tsne_df.to_csv(output_file_path, index=False) #save output to CSV file
        print(f"saved: {output_file_name} in {output_dir}") #print confirmation message

################################################################
# completion message

print("\033[1;32m t-SNE transformation completed! Files saved in:\033[0m", output_base_dir) #print success message
print('use the viz_test_tsne script to determine perplexity for final script') #print hint for next steps
