#this script applies k-means clustering to t-SNE-transformed data
#it assigns clusters for different k values while retaining t-SNE features

from pathlib import Path #for handling file paths
import pandas as pd #for working with dataframes
from sklearn.cluster import KMeans #for clustering

################################################################
#input and output setup  

tsne_dir = Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/tSNE_data") #directory for t-SNE-transformed data
input_dir = Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/working_preprocessed_data") #preprocessed data dir
output_template = Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/k-means_data/K-means_data_tSNE_K={k}") #output format
k_values = [2, 3, 4, 5, 6] #set k values for clustering

################################################################
#loop through t-SNE-transformed files  

for tsne_file in tsne_dir.glob("*.csv"): #iterate over t-SNE-transformed data
    print(f"processing {tsne_file.name}...") #print current file being processed
    
    #match the original preprocessed file using its name
    base_name = tsne_file.stem.replace("_standardized_tSNE-transformed", "").replace("_tSNE-transformed", "") #extract base name

    #ensure `_Phototrophs` is only added if missing
    preprocessed_name = base_name if "_Phototrophs" in base_name else f"{base_name}_Phototrophs.csv"
    preprocessed_path = input_dir / preprocessed_name #full path to corresponding preprocessed file

    #check if the preprocessed file exists  
    if not preprocessed_path.exists():
        print(f"error: preprocessed file not found -> {preprocessed_path}") #print error message
        continue #skip to next file

    tsne_df = pd.read_csv(tsne_file) #load t-SNE-transformed data
    prepro_df = pd.read_csv(preprocessed_path) #load original preprocessed data

    #store original shape for debugging  
    original_shape = prepro_df.shape  
    print(f"original shape: {original_shape}") #print initial dataframe shape
    
    #filter out rows where FL2-H is too low  
    mask = prepro_df['FL2-H'] > 0.1 #create boolean mask for filtering

    #apply mask only to matching indexes to avoid unalignable boolean series error
    filtered_tsne_df = tsne_df.loc[mask.index[mask]] #filter only valid indices

    #store new shape after filtering  
    new_shape = filtered_tsne_df.shape  
    print(f"after filtering: {new_shape}") #print new dataframe shape
    
    ################################################################
    #apply k-means clustering for different k values  

    for k in k_values: #loop through k values
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10) #initialize k-means model
        clusters = kmeans.fit_predict(filtered_tsne_df) #assign clusters

        #set all rows to -1 cluster initially  
        prepro_df['Cluster'] = -1 #assign default value for all rows

        #update only filtered rows with their assigned cluster  
        prepro_df.loc[filtered_tsne_df.index, 'Cluster'] = clusters #apply cluster labels
        
        #keep t-SNE columns in final output  
        prepro_df[['tSNE-1', 'tSNE-2']] = tsne_df.iloc[:, :2] #retain t-SNE columns
        
        #define output directory for the current k value  
        output_dir = output_template.with_name(output_template.name.format(k=k)) #replace {k} with actual k value
        output_dir.mkdir(parents=True, exist_ok=True) #ensure the folder exists
        
        #construct output filename while keeping tSNE in the suffix  
        output_filename = f"{preprocessed_path.stem}_tSNE_kmeans={k}.csv" #modify filename to include k value
        output_file_path = output_dir / output_filename #define full output path

        #save results  
        prepro_df.to_csv(output_file_path, index=False) #save dataframe as csv file
        print(f"k-means (k={k}) done for '{tsne_file.name}'. saved to: {output_file_path}") #print confirmation message

################################################################
#completion message  

print("\033[1;32mall k-means clustering tasks finished! files are saved.\033[0m") #print success message
