#this script visualizes K-Means clustered data
#it allows users to select a file, choose parameters, and generate scatter plots

from pathlib import Path #for handling file operations
import pandas as pd #for data manipulation
import matplotlib.pyplot as plt #for plotting
import numpy as np #for numerical operations
import random #for random file selection
from matplotlib.colors import to_rgba, Normalize #for colour mapping

################################################################
#set directories  

main_dir = Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/k-means_data") #path to clustering results

################################################################
#choose clustering method  

method_list = ["K-means", "PCA K-means", "t-SNE K-means", "UMAP K-means"] #list of available methods

def pick_method():
    #Allows the user to select a clustering method
    print("\navailable methods:")
    for i, name in enumerate(method_list, start=1): #print method options
        print(f"{i}. {name}")
    choice = input("pick a method (1-4): ").strip()
    return method_list[int(choice) - 1] if choice.isdigit() and 1 <= int(choice) <= len(method_list) else "K-means" #default to K-means

################################################################
#choose available k values for selected method  

def pick_k_value(method):
    #Lists available K values for the chosen method and allows user selection
    method_prefix = {
        "K-means": "K-means_data_K=",
        "PCA K-means": "K-means_data_PCA_K=",
        "t-SNE K-means": "K-means_data_tSNE_K=",
        "UMAP K-means": "K-means_data_UMAP_K="
    }.get(method, "K-means_data_K=") #default to K-means if method is invalid

    #find directories matching selected method
    k_folders = [folder for folder in main_dir.iterdir() if folder.is_dir() and folder.name.startswith(method_prefix)]
    
    if not k_folders:
        print("no valid K folders found.")
        return None

    k_values = sorted([folder.name.split("_K=")[-1] for folder in k_folders]) #extract k values
    print("\navailable k values:")
    for i, k in enumerate(k_values, start=1): #display options
        print(f"{i}. K={k}")
    choice = input("pick a k value: ").strip()
    return k_values[int(choice) - 1] if choice.isdigit() and 1 <= int(choice) <= len(k_values) else k_values[0] #default to first k

################################################################
#choose file  

def select_file(folder_path):
    #Allows user to select a file manually or pick one randomly
    if not folder_path.exists():
        print(f"folder does not exist: {folder_path}") #debugging output
        return None
    
    csv_files = list(folder_path.glob("*.csv")) #get list of csv files
    if not csv_files:
        print(f"no files found in {folder_path}") #debugging output
        return None

    print("\n1. choose a specific file")
    print("2. pick a random file")
    choice = input("choose (1 or 2): ")

    if choice == "2":
        return random.choice(csv_files) #return random file

    for i, file in enumerate(csv_files, 1): #list available files
        print(f"{i}. {file.name}")
    while True:
        try:
            return csv_files[int(input("file number: ")) - 1]
        except (ValueError, IndexError):
            print("invalid choice. try again.")

################################################################
#select parameters for visualization  

def select_params(data, method_used):
    #Allows the user to choose predefined parameters or manually select from available columns
    print("\n1. FSC-H vs SSC-H")
    print("2. FL3-H vs FL2-H")
    print("3. custom")
    
    reduced_options = ["PCA K-means", "t-SNE K-means", "UMAP K-means"]
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
        if method_used == "PCA K-means" and {"PC1", "PC2"}.issubset(data.columns):
            return "PC1", "PC2", False
        elif method_used == "t-SNE K-means" and {"tSNE-1", "tSNE-2"}.issubset(data.columns):
            return "tSNE-1", "tSNE-2", False
        elif method_used == "UMAP K-means" and {"UMAP-1", "UMAP-2"}.issubset(data.columns):
            return "UMAP-1", "UMAP-2", False

    return "FSC-H", "SSC-H", True

################################################################
#visualize the data  

def visualize(file, x, y, log_scale):
    #Generates a scatter plot of K-Means clustered data
    data = pd.read_csv(file)

    if log_scale:
        data[x] = np.log10(data[x].replace(0, np.nan))
        data[y] = np.log10(data[y].replace(0, np.nan))
    data = data.dropna(subset=[x, y])

    noise = data[data["Cluster"] == -1] #filter noise
    clusters = data[data["Cluster"] != -1] #filter species

    unique_clusters = sorted(clusters["Cluster"].unique()) #get unique clusters

    #assign colour mapping using Turbo colormap
    cmap = plt.get_cmap("turbo")
    norm = Normalize(vmin=min(unique_clusters), vmax=max(unique_clusters))
    cluster_colors = {cluster: cmap(norm(cluster)) for cluster in unique_clusters}

    #define figure
    plt.figure(figsize=(8, 6))

    #plot noise points
    if not noise.empty:
        plt.scatter(noise[x], noise[y], c="gray", alpha=0.3, label="Noise", s=10)

    #plot clusters
    for cluster in unique_clusters:
        cluster_points = clusters[clusters["Cluster"] == cluster] #filter species data
        plt.scatter(cluster_points[x], cluster_points[y], color=cluster_colors[cluster], label=f"Cluster {cluster}", alpha=0.8, s=15)

    #set plot title
    plt.title(f"K-Means Clustering: {file.stem}") 
    plt.xlabel(f"log10({x})" if log_scale else x) 
    plt.ylabel(f"log10({y})" if log_scale else y) 
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Clusters") 
    plt.tight_layout() 
    plt.show() 

################################################################
#main script  

if __name__ == "__main__":
    print("k-means cluster visualization") 

    method_chosen = pick_method() #select clustering method
    k_value = pick_k_value(method_chosen) #select k value

    if not k_value:
        print("no valid k values found. exiting.") 
        exit()

    #construct correct folder path
    method_prefix = {
        "K-means": "K-means_data_K=",
        "PCA K-means": "K-means_data_PCA_K=",
        "t-SNE K-means": "K-means_data_tSNE_K=",
        "UMAP K-means": "K-means_data_UMAP_K="
    }[method_chosen]

    folder_path = main_dir / f"{method_prefix}{k_value}"
    
    file = select_file(folder_path) #select file

    if file:
        print(f"selected file: {file.name}") 
        data = pd.read_csv(file)
        x, y, log_scale = select_params(data, method_chosen) #allow reduced dimensions selection
        visualize(file, x, y, log_scale) #generate visualization
