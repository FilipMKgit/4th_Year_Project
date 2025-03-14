#this script visualizes clustered data from GMM results
#it allows users to select a clustering method, choose a file, and generate scatter plots

from pathlib import Path #for handling file operations
import pandas as pd #for data manipulation
import numpy as np #for numerical operations
import random #for selecting random files
import matplotlib.pyplot as plt #for plotting
from sklearn.metrics import silhouette_score #for clustering evaluation

################################################################
#set directories  

main_dir = Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/GMM_data") #path to GMM results

################################################################
#choose clustering method  

method_list = ["GMM", "PCA GMM", "t-SNE GMM", "UMAP GMM"] #list of available methods

def pick_method():
    #allows the user to select a clustering method
    print("\navailable methods:")
    for i, name in enumerate(method_list, start=1): #print method options
        print(f"{i}. {name}")
    choice = input("pick a method (1-4): ").strip()
    return method_list[int(choice) - 1] if choice.isdigit() and 1 <= int(choice) <= len(method_list) else "GMM" #default to GMM

################################################################
#choose file  

def pick_file(folder_path):
    #allows the user to select a file manually or pick one randomly.
    csv_files = list(folder_path.glob("*.csv")) #get list of csv files
    if not csv_files:
        print(f"no files found in {folder_path}") #debugging output
        return None

    print("\n1. choose a specific file") #option to select manually
    print("2. pick a random file") #option to select randomly
    user_choice = input("choose (1 or 2): ").strip()

    if user_choice == "2":
        return random.choice(csv_files) #return random file

    for i, file in enumerate(csv_files, 1): #list available files
        print(f"{i}. {file.name}")
    while True:
        try:
            return csv_files[int(input("file number: ")) - 1]
        except (ValueError, IndexError):
            print("invalid choice. try again.") #ask again

################################################################
#choose plot parameters  

def pick_plot_columns(df, method_used):
    #allows the user to choose predefined parameters or manually select from available columns
    print("\nchoose plot parameters:")
    print("1. FSC-H vs SSC-H") #first option
    print("2. FL3-H vs FL2-H") #second option
    print("3. custom selection") #manual selection

    reduced_options = ["PCA GMM", "t-SNE GMM", "UMAP GMM"]
    if method_used in reduced_options:
        print("4. reduced dimensions") #only show this for transformed data

    user_input = input("enter number: ").strip()

    if user_input == "1":
        return "FSC-H", "SSC-H", True
    elif user_input == "2":
        return "FL3-H", "FL2-H", True
    elif user_input == "3":
        column_names = [col for col in df.columns if col not in ["Time", "Cluster"]] #exclude unnecessary columns
        for i, col in enumerate(column_names, start=1):
            print(f"{i}. {col}")
        try:
            x_idx = int(input("x-axis column number: ")) - 1
            y_idx = int(input("y-axis column number: ")) - 1
            return column_names[x_idx], column_names[y_idx], True
        except (ValueError, IndexError):
            return "FSC-H", "SSC-H", True
    elif user_input == "4" and method_used in reduced_options:
        if method_used == "PCA GMM" and {"PC1", "PC2"}.issubset(df.columns):
            return "PC1", "PC2", False
        elif method_used == "t-SNE GMM" and {"tSNE-1", "tSNE-2"}.issubset(df.columns):
            return "tSNE-1", "tSNE-2", False
        elif method_used == "UMAP GMM" and {"UMAP-1", "UMAP-2"}.issubset(df.columns):
            return "UMAP-1", "UMAP-2", False

    return "FSC-H", "SSC-H", True

################################################################
#silhouette score calculation  

def calc_silhouette(path_to_file):
    #computes the silhouette score for the selected file
    df = pd.read_csv(path_to_file) #load data
    df_filtered = df[df['Cluster'] != -1] #remove unclustered points

    if df_filtered.empty: #check if valid data exists
        print(f"{path_to_file.name}: not enough clusters to compute silhouette score.") #print warning
        return

    cluster_labels = df_filtered['Cluster'] #extract cluster labels
    data_points = df_filtered.drop(columns=['Cluster']) #remove label column
    silhouette = silhouette_score(data_points, cluster_labels) #compute score
    print(f"{path_to_file.name}: silhouette score = {round(silhouette, 2)}") #print result

################################################################
#plot clusters  

def plot_clusters(file_path, x, y, log_scale):
    #generates a scatter plot of GMM clustered data
    df = pd.read_csv(file_path) #load data
    if log_scale: #apply log transform if needed
        df[x] = np.log10(df[x].replace(0, np.nan)) 
        df[y] = np.log10(df[y].replace(0, np.nan)) 
    df.dropna(subset=[x, y], inplace=True) #remove NaNs
    unique_clusters = sorted(df['Cluster'].unique()) #get unique clusters

    #generate colors using turbo colormap
    turbo_cmap = plt.get_cmap('turbo', len(unique_clusters))
    color_mapping = {-1: 'gray'} #assign gray for noise points

    for i, cluster_id in enumerate(unique_clusters):
        if cluster_id != -1:
            color_mapping[cluster_id] = turbo_cmap(i / len(unique_clusters))

    plt.figure(figsize=(8, 6)) #define figure size
    for cluster in unique_clusters: #plot each cluster
        subset_df = df[df['Cluster'] == cluster] #filter data per cluster
        plt.scatter(subset_df[x], subset_df[y], color=color_mapping[cluster], label=f"Cluster {cluster}", alpha=0.7, s=15) 

    plt.xlabel(x) #set x-axis label
    plt.ylabel(y) #set y-axis label
    plt.title(file_path.name) #set plot title
    plt.legend(title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left') #add legend
    plt.tight_layout() #adjust layout
    plt.show() #display plot

################################################################
#execution of script  

if __name__ == "__main__":
    method_chosen = pick_method() #select clustering method

    #mapping method names to folder structure
    folder_map = {
        "GMM": "gmm_data",
        "PCA GMM": "gmm_data_PCA",
        "t-SNE GMM": "gmm_data_tSNE",
        "UMAP GMM": "gmm_data_UMAP"
    }

    folder_path = main_dir / folder_map.get(method_chosen, "") #get path to method folder
    file = pick_file(folder_path) #select file

    if file: #if a file is selected
        print(f"selected file: {file.name}") #print selected file
        df_data = pd.read_csv(file) #load data
        calc_silhouette(file) #compute silhouette score
        x_col, y_col, log_scale = pick_plot_columns(df_data, method_chosen) #user selects axes for plotting
        plot_clusters(file, x_col, y_col, log_scale) #plot results
