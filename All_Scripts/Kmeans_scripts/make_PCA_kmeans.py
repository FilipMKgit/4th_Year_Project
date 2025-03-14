#this script applies k-means clustering to PCA-transformed data
#it assigns clusters for different k values while retaining PCA features

from pathlib import Path #for handling file paths
import pandas as pd #for working with dataframes
from sklearn.cluster import KMeans #for clustering

################################################################
#input and output setup

pca_dir = Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/PCA_data") #directory for PCA-transformed data
input_dir = Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/working_preprocessed_data") #preprocessed data dir
output_template = Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/k-means_data/K-means_data_PCA_K={k}") #output format
k_values = [2, 3, 4, 5, 6] #set k values for clustering

################################################################
#loop through PCA-transformed files

for pca_file in pca_dir.glob("*.csv"): #iterate over PCA-transformed data
    print(f"processing {pca_file.name}...") #print current file being processed
    
    #match the original preprocessed file using its name
    preprocessed_name = pca_file.stem.replace("_standardized_PCA-transformed", "") + ".csv" #extract original filename
    preprocessed_path = input_dir / preprocessed_name #full path to corresponding preprocessed file

    pca_df = pd.read_csv(pca_file) #load PCA-transformed data
    prepro_df = pd.read_csv(preprocessed_path) #load original preprocessed data

    #store original shape for debugging
    original_shape = prepro_df.shape  
    print(f"original shape: {original_shape}") #print initial dataframe shape
    
    #filter out rows where FL2-H is too low
    mask = prepro_df['FL2-H'] > 0.1 #create boolean mask for filtering
    filtered_pca_df = pca_df[mask] #apply filter to PCA data

    #store new shape after filtering
    new_shape = filtered_pca_df.shape  
    print(f"after filtering: {new_shape}") #print new dataframe shape
    
    ################################################################
    #apply k-means clustering for different k values

    for k in k_values: #loop through k values
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10) #initialize k-means model
        clusters = kmeans.fit_predict(filtered_pca_df) #assign clusters

        #set all rows to -1 cluster initially
        prepro_df['Cluster'] = -1 #assign default value for all rows

        #update only filtered rows with their assigned cluster
        prepro_df.loc[filtered_pca_df.index, 'Cluster'] = clusters #apply cluster labels
        
        #keep PCA columns in final output
        pca_columns = [col for col in pca_df.columns if "PC" in col] #find PCA columns
        prepro_df[pca_columns] = pca_df[pca_columns] #add them back to the dataframe

        #define output directory for the current k value
        output_dir = output_template.with_name(output_template.name.format(k=k)) #replace {k} with actual k value
        output_dir.mkdir(parents=True, exist_ok=True) #ensure the folder exists
        
        #construct output filename  
        output_filename = f"{preprocessed_path.stem}_PCA__kmeans={k}.csv" #modify filename to include k value
        output_file_path = output_dir / output_filename #define full output path

        #save results
        prepro_df.to_csv(output_file_path, index=False) #save dataframe as csv file
        print(f"k-means (k={k}) done for '{pca_file.name}'. saved to: {output_file_path}") #print confirmation message

################################################################
#completion message

print("\033[1;32mall k-means clustering tasks finished! files are saved.\033[0m") #print success message
