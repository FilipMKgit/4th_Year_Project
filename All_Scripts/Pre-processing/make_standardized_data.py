#This code standardizes annotated files with noise for later use

from pathlib import Path #for handling file paths
import pandas as pd #for data manipulation
from sklearn.preprocessing import StandardScaler #standardization applied

################################################################
#set input and output directories  

input_dir = Path(r"C:\Users\filip\OneDrive\Desktop\4YR_PROJ\work_data\annotated_data_with_noise") #defines input directory
output_dir = Path(r"C:\Users\filip\OneDrive\Desktop\4YR_PROJ\work_data\working_data_standardized") #defines output directory
output_dir.mkdir(parents=True, exist_ok=True) #creates directory if it doesn't exist

new_suffix = "_standardized" #defines the suffix for output files

################################################################
#loop thorugh all files in input dir

for file_path in input_dir.glob("*.csv"): #looks for files with .csv extension and constructs a full file path for each file for pandas
    df = pd.read_csv(file_path) #pandas reads the full file path and formats it into a dataframe (df)
    original_column_order = df.columns.tolist() #stores original column order for later
    columns_to_standardize = ["FSC-A", "FSC-H", "SSC-A", "SSC-H", "FL1-A", "FL1-H", "FL2-A", "FL2-H", "FL3-A", "FL3-H", "FL4-A", "FL4-H"] 
    #select parameters to standardize here (most important are FSC-H, SSC-H, FL2-H, FL3-H)
    columns_to_standardize = [col for col in columns_to_standardize if col in df.columns] #filters only columns present in the dataframe
    scaler = StandardScaler() #standardizes the selected columns with StandardScaler()
    df[columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize]) #applies standardization to selected columns
    df = df[original_column_order] #restores the original column order

    #output file editing 
    output_file_name = file_path.stem.replace("_annotated_with_noise", "") + new_suffix + ".csv" #removes the previous suffix and adds new one
    output_path = output_dir / output_file_name #constructs full output file path
    df.to_csv(output_path, index=False) #saves the standardized dataframe to a new file

    #operation progress is printed
    print(f"standardized {file_path.name} to {output_file_name} and saved in {output_dir}") #prints confirmation of file processing  

################################################################
#completion message  
print("\033[1;32mStandardization completed! output can be found in:\033[0m", output_dir) #prints final completion message