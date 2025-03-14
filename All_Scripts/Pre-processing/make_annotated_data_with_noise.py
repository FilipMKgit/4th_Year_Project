#this script compares annotated data files with their corresponding raw data files
#in each file it compares rows and rows exclusive to raw data files are labelled as 'Noise' in cluster '-1'
#it then adds the noise rows into the annotated data files and saves them to annotated_data_with_noise

from pathlib import Path #for handling file operations
import pandas as pd #for data manipulation

################################################################
#defines input and output directories

raw_data_dir = Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/raw_data/phototrophs") #where raw data is stored
annotated_data_dir = Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/annotated_data") #where annotated files are stored
output_dir = Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/annotated_data_with_noise") #where new files will be saved

output_dir.mkdir(parents=True, exist_ok=True) #make sure output folder exists

################################################################
#process each annotated file

for annotated_file in annotated_data_dir.iterdir(): #loop through annotated files
    if not annotated_file.name.endswith("_annotated.csv"): #skip unrelated files
        continue
    
    prefix = annotated_file.stem.replace("_annotated", "") #extract base name
    raw_file = raw_data_dir / f"{prefix}_Phototrophs.csv" #expected raw data file
    output_file = output_dir / f"{prefix}_annotated_with_noise.csv" #output file path
    
    if not raw_file.exists(): #check if raw data file exists
        print(f"raw data file missing for {annotated_file.name}, skipping.") #print warning and skip file
        continue
    
    df_annotated = pd.read_csv(annotated_file) #load annotated data
    df_raw = pd.read_csv(raw_file) #load raw phototrophs data
    
    #get common columns (excluding 'Cluster' and 'Label')
    common_columns = [col for col in df_raw.columns if col in df_annotated.columns] #find columns present in both datasets
    
    #find raw data rows that are not in annotated data
    df_remain = df_raw.merge(df_annotated[common_columns], how='left', indicator=True) #merge to find unmatched rows
    df_remain = df_remain[df_remain['_merge'] == 'left_only'].drop(columns=['_merge']) #filter out only unmatched rows
    
    df_remain['Cluster'] = -1 #assign as noise
    df_remain['Label'] = 'Noise' #label the noise data
    
    df_final = pd.concat([df_annotated, df_remain], ignore_index=True) #combine annotated and noise data
    
    df_final.to_csv(output_file, index=False) #save final dataset
    print(f"processed '{annotated_file.name}', output saved to '{output_file}'.") #print confirmation message

################################################################
#completion message

print("\033[1;32mall files processed successfully!\033[0m") #print success message
print("(execute the 'viz_annotated_data_with_noise'/'make_annotated_data' scripts to visualize the outputs)") #print hint for visualization

