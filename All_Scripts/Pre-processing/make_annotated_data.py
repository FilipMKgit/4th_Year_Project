#this script combines gate files one by one to produce annotated files

from pathlib import Path #for handling file operations
import pandas as pd #for working with csv files

################################################################
#defines input and output directories

input_dir = Path(r"C:\Users\filip\OneDrive\Desktop\4YR_PROJ\raw_data\gates") #where raw gate files are stored
output_dir = Path(r"C:\Users\filip\OneDrive\Desktop\4YR_PROJ\work_data\annotated_data") #where annotated data will be saved
output_dir.mkdir(parents=True, exist_ok=True) #make sure output folder exists

################################################################
#define gate suffixes and corresponding cluster numbers

g_suffix = ["_Crypto", "_Nano1", "_Nano2", "_PEuk1", "_PEuk2", "_Syn"] #plankton groups

################################################################
#process each `_Phototrophs` file

for file in input_dir.iterdir(): #loop through all files in input folder
    if not file.name.endswith("_Phototrophs.csv"): #skip files that aren't phototroph data
        continue
    
    filepath = input_dir / file #full path to the phototroph file
    df_m = pd.read_csv(filepath) #load phototroph data into a dataframe
    combined_gates = pd.DataFrame() #empty dataframe to store merged gate data
    prefix = file.stem.replace("_Phototrophs", "") #get prefix (base file name)
    
    #loop through each gate file and match with main phototroph file
    for cluster_num, suffix in enumerate(g_suffix, start=1): #assign cluster numbers starting from 1
        gate_filename = f"{prefix}{suffix}.csv" #construct expected gate file name
        gate_filepath = input_dir / gate_filename #full path to the gate file
        
        if gate_filepath.exists(): #check if gate file is available
            df_g = pd.read_csv(gate_filepath) #load gate data
            df_g["Cluster"] = cluster_num #assign a cluster number
            df_g["Label"] = suffix[1:] #remove the leading underscore in label
            
            #filter out duplicates if any
            if not combined_gates.empty: #only check if there's already data
                df_g = df_g[~df_g.apply(tuple, axis=1).isin(combined_gates.apply(tuple, axis=1))] #remove duplicates
            
            #append gate data to the main dataframe
            combined_gates = pd.concat([combined_gates, df_g], ignore_index=True) #merge new data with previous gates
    
    #if no gate data was found, skip saving
    if combined_gates.empty: #check if merged dataframe is empty
        print(f"no gate files found for {prefix}. skipping.") #print message and skip this file
        continue
    
    #remove any remaining duplicates
    combined_gates = combined_gates.drop_duplicates() #drop duplicate rows if any
    
    #save the final annotated file
    output_filename = f"{prefix}_annotated.csv" #construct output filename
    combined_gates.to_csv(output_dir / output_filename, index=False) #save to output folder
    print(f"annotated data saved for '{file.name}' at '{output_dir}'.") #print save confirmation

################################################################
#completion message

print("\033[1;32m Annotation Completed! \033[0m") #print success message
print("(run the 'make_annotated_data_with_noise' script to generate annotated files that include noise)") #print next important step