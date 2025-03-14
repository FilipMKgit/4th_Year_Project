#this script visualizes clustered data from t-SNE test files
#it allows users to select a perplexity folder, choose a file, and generate scatter plots

from pathlib import Path #for handling file operations
import pandas as pd #for data manipulation
import numpy as np #for numerical operations
import random #for selecting random files
import matplotlib.pyplot as plt #for plotting
from matplotlib.colors import to_rgba #for color mapping

################################################################
#set directories  

main_dir = Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/tSNE_test_data") #path to t-SNE test data

################################################################
#choose a perplexity folder  

def pickPerplexityFolder():
    #allows the user to select a perplexity folder
    perplexity_folders = sorted([f for f in main_dir.iterdir() if f.is_dir() and f.name.startswith("tSNE_test_p=")], key=lambda x: int(x.name.split("=")[-1])) #filter and sort folders
    print("\navailable t-SNE perplexity values:")
    for i, folder in enumerate(perplexity_folders, start=1): #display available folders
        print(f"{i}. {folder.name}")
    choice = input("pick a perplexity folder: ").strip()
    return perplexity_folders[int(choice) - 1] if choice.isdigit() and 1 <= int(choice) <= len(perplexity_folders) else perplexity_folders[0] #default to first folder

################################################################
#choose a file  

def pickFile(folder_path):
    #allows the user to select a file manually or pick one randomly
    csvFiles = list(folder_path.glob("*.csv")) #get all csv files in folder
    print("\n1. choose a specific file") #option to select manually
    print("2. pick a random file") #option to select randomly
    user_choice = input("enter 1 or 2: ").strip()

    if user_choice == "2":
        return random.choice(csvFiles) #return random file

    for i, file in enumerate(csvFiles, start=1): #list all files
        print(f"{i}. {file.name}")
    selection = input("enter file number: ").strip()
    return csvFiles[int(selection) - 1] if selection.isdigit() and 1 <= int(selection) <= len(csvFiles) else csvFiles[0] #default to first file

################################################################
#plot t-SNE clusters with mapped labels  

def plot_tsne(filePath):
    #reads the selected file and generates a t-SNE scatter plot with cluster colors
    df = pd.read_csv(filePath) #load data

    #check if required columns exist
    if not {'tSNE1', 'tSNE2', 'Cluster', 'Label'}.issubset(df.columns): #validate required columns
        print("error: the selected file does not contain the necessary t-SNE transformed data.") #print error message
        return #exit function

    #create cluster-to-label mapping
    cluster_mapping = dict(zip(df["Cluster"], df["Label"])) #map cluster numbers to species labels

    #separate noise and species
    noise = df[df["Cluster"] == -1] #filter noise points
    species = df[df["Cluster"] != -1] #filter species points

    #assign colors for species using the turbo colormap
    unique_clusters = sorted(species["Cluster"].unique()) #get unique species clusters
    cmap = plt.get_cmap("turbo") #define colourmap
    cluster_colors = {cluster: cmap(i / max(1, len(unique_clusters) - 1)) for i, cluster in enumerate(unique_clusters)} #assign colours

    #plot the data
    plt.figure(figsize=(8, 6)) #define figure size

    #plot noise points
    if not noise.empty:
        plt.scatter(noise["tSNE1"], noise["tSNE2"], c="gray", alpha=0.3, label="Noise", s=10) #plot noise

    #plot species
    for cluster in unique_clusters:
        cluster_points = species[species["Cluster"] == cluster] #filter cluster data
        plt.scatter(cluster_points["tSNE1"], cluster_points["tSNE2"],
            color=to_rgba(cluster_colors[cluster]), label=cluster_mapping.get(cluster, f"Cluster {cluster}"), alpha=0.8, s=15) #plot species

    #customize the plot
    plt.title(f"t-SNE Visualization: {filePath.name}") #set title
    plt.xlabel("tSNE1") #set x-axis label
    plt.ylabel("tSNE2") #set y-axis label
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Species") #add legend
    plt.tight_layout() #adjust layout
    plt.show() #display plot

################################################################
#execution of script  

if __name__ == "__main__":
    perplexity_folder = pickPerplexityFolder() #prompt user to select perplexity folder
    file_name = pickFile(perplexity_folder) #prompt user to select file
    file_path = perplexity_folder / file_name #construct full path

    df_data = pd.read_csv(file_path) #load data
    plot_tsne(file_path) #generate visualization
