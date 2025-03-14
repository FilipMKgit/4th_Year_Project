#this script applies PCA to standardized data
#it retains components that explain 90% of the variance

from pathlib import Path #for handling file operations
import pandas as pd #for data manipulation
from sklearn.decomposition import PCA #for PCA transformation

################################################################
#defines input and output directories

input_dir = Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/working_data_standardized") #where standardized data is stored
output_dir = Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/PCA_data") #where PCA-transformed data will be saved
new_suffix = "_PCA-transformed" #suffix for processed files
output_dir.mkdir(parents=True, exist_ok=True) #make sure output folder exists

################################################################
#process each CSV file in the folder

for file_path in input_dir.glob("*.csv"): #loop through all CSV files
    print(f"processing {file_path.name}...") #debugging print
    
    df = pd.read_csv(file_path) #load file into dataframe
    original_shape = df.shape #store original shape for debugging
    df = df.copy() #make a copy for safety
    
    #initialize PCA model
    pca = PCA(n_components=0.90) #keep components explaining 90% of variance
    
    #apply PCA transformation
    transformed_data = pca.fit_transform(df) #perform PCA on the data
    num_components = pca.n_components_ #store number of retained components
    print(f"original shape: {original_shape}, new shape: {transformed_data.shape}") #debugging info
    
    #generate PCA column names
    pca_columns = [f'PC{i+1}' for i in range(transformed_data.shape[1])] #create PC labels
    
    #convert transformed data into dataframe
    pca_df = pd.DataFrame(transformed_data, columns=pca_columns) #store PCA results

    #construct output filename  
    output_file_name = file_path.stem + new_suffix + ".csv" #append new suffix
    output_file_path = output_dir / output_file_name #define full path
    
    #save PCA-transformed file
    pca_df.to_csv(output_file_path, index=False) #save file to output folder
    print(f"{file_path.name}: {num_components} components retained, saved as {output_file_name}") #progress printed

################################################################
#completion message

print("\033[1;32m PCA transformation completed! Files saved in:\033[0m", output_dir) #print success message
