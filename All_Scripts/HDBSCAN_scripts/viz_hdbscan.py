from pathlib import Path #for handling file operations
import pandas as pd #for working with dataframes
import numpy as np #for numerical operations
import random #for selecting random files
import matplotlib.pyplot as plt #for plotting
from sklearn.metrics import silhouette_score #for clustering evaluation

################################################################
#set directories

main_dir = Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/HDBSCAN_data") #path to HDBSCAN results

################################################################
#choose clustering method

method_list = ["HDBSCAN", "PCA HDBSCAN", "t-SNE HDBSCAN", "UMAP HDBSCAN"]

def pick_method():
    print("\navailable methods:")
    for i, name in enumerate(method_list, start=1): #print method options
        print(f"{i}. {name}")
    choice = input("pick a method (1-4): ").strip()
    return method_list[int(choice) - 1] if choice.isdigit() and 1 <= int(choice) <= len(method_list) else "HDBSCAN" #default to HDBSCAN

################################################################
#choose available files

def pick_file(folder_path):
    csv_files = list(folder_path.glob("*.csv")) #get list of csv files
    if not csv_files:
        print(f"no files found in {folder_path}") #debugging output
        return None

    print("\n1. choose a specific file")
    print("2. pick a random file")
    user_choice = input("choose (1 or 2): ").strip()

    if user_choice == "2":
        return random.choice(csv_files) #return random file

    for i, file in enumerate(csv_files, 1): #list available files
        print(f"{i}. {file.name}")
    while True:
        try:
            return csv_files[int(input("file number: ")) - 1]
        except (ValueError, IndexError):
            print("invalid choice. try again.")

################################################################
#silhouette score calculation

def calc_silhouette(path_to_file):
    df = pd.read_csv(path_to_file) #load data
    df_filtered = df[df['Cluster'] != -1] #remove unclustered points
    if df_filtered.empty:
        print(f"{path_to_file.name}: not enough clusters to compute silhouette score.")
        return
    cluster_labels = df_filtered['Cluster'] #extract cluster labels
    data_points = df_filtered.drop(columns=['Cluster']) #remove label column
    silhouette = silhouette_score(data_points, cluster_labels) #compute score
    print(f"{path_to_file.name}: silhouette score = {round(silhouette, 2)}") 

################################################################
#select parameters for visualization

def select_params(data, method_used):
    print("\n1. FSC-H vs SSC-H")
    print("2. FL3-H vs FL2-H")
    print("3. custom")
    
    reduced_options = ["PCA HDBSCAN", "t-SNE HDBSCAN", "UMAP HDBSCAN"]
    if method_used in reduced_options:
        print("4. reduced dimensions")
    
    choice = input("choose (1, 2, 3, or 4): ")

    if choice == "1":
        return "FSC-H", "SSC-H", True
    elif choice == "2":
        return "FL3-H", "FL2-H", True
    elif choice == "3":
        cols = [col for col in data.columns if col not in ["Time", "Cluster"]] #get valid columns
        for i, col in enumerate(cols, 1):
            print(f"{i}. {col}")
        while True:
            try:
                x = cols[int(input("x-axis parameter: ")) - 1]
                y = cols[int(input("y-axis parameter: ")) - 1]
                return x, y, True
            except (ValueError, IndexError):
                print("invalid selection. try again.")
    elif choice == "4" and method_used in reduced_options:
        if method_used == "PCA HDBSCAN" and {"PC1", "PC2"}.issubset(data.columns):
            return "PC1", "PC2", False
        elif method_used == "t-SNE HDBSCAN" and {"tSNE-1", "tSNE-2"}.issubset(data.columns):
            return "tSNE-1", "tSNE-2", False
        elif method_used == "UMAP HDBSCAN" and {"UMAP-1", "UMAP-2"}.issubset(data.columns):
            return "UMAP-1", "UMAP-2", False

    return "FSC-H", "SSC-H", True

################################################################
#plot clusters
################################################################
#plot clusters

def plot_clusters(file_path, x, y, log_scale):
    df = pd.read_csv(file_path) #load data
    if log_scale: #apply log transform if needed
        df[x] = np.log10(df[x].replace(0, np.nan)) 
        df[y] = np.log10(df[y].replace(0, np.nan)) 
    df.dropna(subset=[x, y], inplace=True) #remove NaNs
    unique_clusters = sorted(df['Cluster'].unique()) #get unique clusters

    # Generate colors using Turbo colormap
    turbo_cmap = plt.get_cmap('turbo', len(unique_clusters))
    color_mapping = {-1: 'gray'}  # Assign gray for noise points

    for i, cluster_id in enumerate(unique_clusters):
        if cluster_id != -1:
            color_mapping[cluster_id] = turbo_cmap(i / len(unique_clusters))

    plt.figure(figsize=(8, 6)) 
    for cluster in unique_clusters: #plot each cluster
        subset_df = df[df['Cluster'] == cluster] 
        plt.scatter(subset_df[x], subset_df[y], color=color_mapping[cluster], label=f"Cluster {cluster}", alpha=0.7, s=15) 
    
    plt.xlabel(x) 
    plt.ylabel(y) 
    plt.title(file_path.name) 
    plt.legend(title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left') 
    plt.tight_layout() 
    plt.show() 


################################################################
#execution of script

if __name__ == "__main__":
    method_chosen = pick_method() #select clustering method

    #mapping method names to folder structure
    folder_map = {
        "HDBSCAN": "hdbscan_data",
        "PCA HDBSCAN": "hdbscan_data_PCA",
        "t-SNE HDBSCAN": "hdbscan_data_tSNE",
        "UMAP HDBSCAN": "hdbscan_data_UMAP"
    }

    folder_path = main_dir / folder_map.get(method_chosen, "") #get path to method folder
    file = pick_file(folder_path) #select file

    if file:
        print(f"selected file: {file.name}")
        df_data = pd.read_csv(file) #load data
        calc_silhouette(file) #compute silhouette score
        x, y, log_scale = select_params(df_data, method_chosen) #user selects axes for plotting
        plot_clusters(file, x, y, log_scale) #plot results
