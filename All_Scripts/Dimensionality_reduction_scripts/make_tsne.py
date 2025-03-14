#this script applies t-SNE to standardized data to reduce dimensionality

from pathlib import Path #for handling file operations
import pandas as pd #for data manipulation
from sklearn.manifold import TSNE #for t-SNE transformation

########################################################################
#defines input and output directories

input_dir = Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/working_data_standardized") #where standardized data is stored
output_dir = Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/tSNE_data") #where t-SNE transformed data will be saved
new_suffix = "_tSNE-transformed" #suffix for processed files
output_dir.mkdir(parents=True, exist_ok=True) #make sure output folder exists

#columns to use for t-SNE transformation
tsne_columns = [  "FSC-A", "FSC-H", "SSC-A", "SSC-H", "FL1-A", "FL1-H", "FL2-A", "FL2-H",  "FL3-A", "FL3-H", "FL4-A", "FL4-H"]
#list of features used for t-SNE

print("executing t-SNE transformation...") #execution message

#########################################################################
#process each CSV file in the folder

for file_path in input_dir.glob("*.csv"): #loop through all CSV files
    print(f"processing {file_path.name}...") #print current file being processed
    
    df = pd.read_csv(file_path) #read CSV file
    df = df.copy() #copy dataframe
    original_shape = df.shape #store original shape for debugging
    
    #drop missing values and keep only selected columns
    df_tsne_input = df[tsne_columns].dropna() #remove NaNs from selected features
    new_shape = df_tsne_input.shape #store new shape after dropping NaNs
    print(f"original shape: {original_shape}, after NaN removal: {new_shape}") #print shape changes
    
    #initialize t-SNE model
    tsne = TSNE(n_components=2, perplexity=5, learning_rate=400, random_state=42) #<------- t-SNE parameters
    
    #apply t-SNE transformation
    transformed_data = tsne.fit_transform(df_tsne_input) #perform t-SNE
    
    #convert transformed data into dataframe
    tsne_df = pd.DataFrame(transformed_data, columns=["tSNE1", "tSNE2"]) #store results in dataframe
    
    #construct output filename  
    output_file_name = f"{file_path.stem}{new_suffix}.csv" #append new suffix
    output_file_path = output_dir / output_file_name #define full path
    
    #save transformed file
    tsne_df.to_csv(output_file_path, index=False) #save output to CSV file
    print(f"{file_path.name} saved as {output_file_name}") #print confirmation message

########################################################################
#completion message

print("\033[1;32m t-SNE transformation completed! Files saved in:\033[0m", output_dir) #print success message
