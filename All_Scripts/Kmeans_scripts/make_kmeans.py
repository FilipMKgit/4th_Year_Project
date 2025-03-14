#this script applies k-means clustering to preprocessed data

from pathlib import Path #for handling file paths
import pandas as pd #for working with dataframes
from sklearn.cluster import KMeans #for clustering

################################################################
#input and output setup

input_dir = Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/working_preprocessed_data") #where preprocessed data is stored
output_template = Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/k-means_data/K-means_data_K={k}") #output directory pattern
k_values = [2, 3, 4, 5, 6] #set k values for clustering

################################################################
#loop through each CSV file in the directory

for file_path in input_dir.glob("*.csv"): #iterate over all csv files in directory
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
    
    ################################################################
    #apply k-means clustering for different k values

    for k in k_values: #loop through k values
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10) #initialize k-means model
        clusters = kmeans.fit_predict(filtered_df) #assign clusters to data points

        #set all rows to -1 cluster initially
        df['Cluster'] = -1 #assign default value for all rows

        #update only filtered rows with their assigned cluster
        df.loc[filtered_df.index, 'Cluster'] = clusters #apply cluster labels
        
        #define output directory for the current k value
        output_dir = output_template.with_name(output_template.name.format(k=k)) #replace {k} with actual k value
        output_dir.mkdir(parents=True, exist_ok=True) #ensure the folder exists
        
        #construct output filename  
        output_filename = f"{file_path.stem}_kmeans={k}.csv" #modify filename to include k value
        output_file_path = output_dir / output_filename #define full output path

        #save results
        df.to_csv(output_file_path, index=False) #save dataframe as csv file
        print(f"k-means (k={k}) done for '{file_path.name}'. saved to: {output_file_path}") #print confirmation message

################################################################
#completion message

print("\033[1;32mall k-means clustering tasks finished! files are saved.\033[0m") #print success message
