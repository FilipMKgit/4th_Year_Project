#this script applies k-means clustering to UMAP-transformed data
#it assigns clusters for different k values while retaining UMAP features

from pathlib import Path #for handling file paths
import pandas as pd #for working with dataframes
from sklearn.cluster import KMeans #for clustering

################################################################
#input and output setup  

umap_dir = Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/UMAP_data") #directory for UMAP-transformed data
input_dir = Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/working_preprocessed_data") #preprocessed data dir
output_template = Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/k-means_data/K-means_data_UMAP_K={k}") #output format
k_values = [2, 3, 4, 5, 6] #set k values for clustering

################################################################
#loop through UMAP-transformed files  

for umap_file in umap_dir.glob("*.csv"): #iterate over UMAP-transformed data
    print(f"processing {umap_file.name}...") #print current file being processed
    
    #match the original preprocessed file using its name
    base_name = umap_file.stem.replace("_Phototrophs_standardized_UMAP-transformed", "").replace("_standardized_UMAP-transformed", "").replace("_UMAP-transformed", "") #extract base name

    #ensure `_Phototrophs` is always included in the reconstructed filename
    preprocessed_name = f"{base_name}_Phototrophs.csv"
    preprocessed_path = input_dir / preprocessed_name #full path to corresponding preprocessed file

    #check if the preprocessed file exists  
    if not preprocessed_path.exists():
        print(f"error: preprocessed file not found -> {preprocessed_path}") #print error message
        continue #skip to next file

    umap_df = pd.read_csv(umap_file) #load UMAP-transformed data
    prepro_df = pd.read_csv(preprocessed_path) #load original preprocessed data

    #store original shape for debugging  
    original_shape = prepro_df.shape  
    print(f"original shape: {original_shape}") #print initial dataframe shape
    
    #filter out rows where FL2-H is too low  
    mask = prepro_df['FL2-H'] > 0.1 #create boolean mask for filtering

    #apply mask only to matching indexes to avoid unalignable boolean series error
    filtered_umap_df = umap_df.loc[mask.index.intersection(umap_df.index)] #ensure valid indices

    #store new shape after filtering  
    new_shape = filtered_umap_df.shape  
    print(f"after filtering: {new_shape}") #print new dataframe shape
    
    ################################################################
    #apply k-means clustering for different k values  

    for k in k_values: #loop through k values
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10) #initialize k-means model
        clusters = kmeans.fit_predict(filtered_umap_df) #assign clusters

        #set all rows to -1 cluster initially  
        prepro_df['Cluster'] = -1 #assign default value for all rows

        #update only filtered rows with their assigned cluster  
        prepro_df.loc[filtered_umap_df.index, 'Cluster'] = clusters #apply cluster labels
        
        #keep UMAP columns in final output  
        prepro_df[['UMAP-1', 'UMAP-2']] = umap_df.iloc[:, :2] #retain UMAP columns
        
        #define output directory for the current k value  
        output_dir = output_template.with_name(output_template.name.format(k=k)) #replace {k} with actual k value
        output_dir.mkdir(parents=True, exist_ok=True) #ensure the folder exists
        
        #construct output filename while keeping UMAP in the suffix  
        output_filename = f"{preprocessed_path.stem}_UMAP_kmeans={k}.csv" #modify filename to include k value
        output_file_path = output_dir / output_filename #define full output path

        #save results  
        prepro_df.to_csv(output_file_path, index=False) #save dataframe as csv file
        print(f"k-means (k={k}) done for '{umap_file.name}'. saved to: {output_file_path}") #print confirmation message

################################################################
#completion message  

print("\033[1;32mall k-means clustering tasks finished! files are saved.\033[0m") #print success message
