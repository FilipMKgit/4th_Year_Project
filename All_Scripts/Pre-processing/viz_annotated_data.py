#this script visualizes clustered data from annotated cytometry files
#it allows users to select a file, choose parameters, and generate scatter plots

from pathlib import Path #for handling file operations
import pandas as pd #for data manipulation
import matplotlib.pyplot as plt #for plotting
import numpy as np #for numerical operations
import random #for random file selection
from matplotlib.colors import to_rgba #for colour mapping
import re #for extracting sample ID

################################################################
#define directory

output_dir = Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/annotated_data") #path to data folder

#cluster mapping (based on original gate suffixes)
cluster_mapping = {1: "Crypto", 2: "Nano1", 3: "Nano2", 4: "PEuk1", 5: "PEuk2", 6: "Syn",-1: "Noise"} #map numeric clusters to species names

################################################################
#select a file (specific or random)

def select_file():
    #Allows the user to select a file manually or choose a random one
    files = [f for f in output_dir.iterdir() if f.suffix == ".csv"] #get csv files
    if not files: #check if any files exist
        print("no files found in the specified folder.") #print warning
        return None #return None if no files found

    print("1. specify a file") #option to manually select a file
    print("2. random file") #option to randomly pick one
    choice = input("choose (1 or 2): ") #ask user for input

    if choice == "2": #if random is chosen
        return random.choice(files) #return a random file
    elif choice == "1": #if manual selection is chosen
        for i, file in enumerate(files, 1): #list all files
            print(f"{i}. {file.name}") 
        while True: #keep asking until valid input
            try:
                return files[int(input("file number: ")) - 1] #return selected file
            except (ValueError, IndexError): #catch invalid input
                print("invalid choice. try again.") #ask again

################################################################
#extract sample ID from filename

def extract_sample_id(filename):
    #Extracts sample ID from filename using regex
    match = re.search(r"(A\d{2} S\d{2} N\d{2})", filename) #search for pattern
    return match.group(1) if match else "Unknown Sample" #return sample ID if found

################################################################
#select parameters for visualization

def select_params(data):
    #Allows the user to choose predefined parameters or manually select from available columns
    print("1. FSC-H vs SSC-H") #first option
    print("2. FL3-H vs FL2-H") #second option
    print("3. custom") #manual selection
    choice = input("choose (1, 2, or 3): ") #ask user for input

    if choice == "1": #predefined option 1
        return "FSC-H", "SSC-H" 
    elif choice == "2": #predefined option 2
        return "FL3-H", "FL2-H"
    
    cols = [col for col in data.columns if col not in ["Time", "Cluster"]] #get valid columns
    for i, col in enumerate(cols, 1): #list available columns
        print(f"{i}. {col}") 
    while True: #keep asking until valid input
        try:
            x = cols[int(input("x-axis parameter: ")) - 1] #get x-axis parameter
            y = cols[int(input("y-axis parameter: ")) - 1] #get y-axis parameter
            return x, y #return chosen parameters
        except (ValueError, IndexError): #catch invalid input
            print("invalid selection. try again.") #ask again

################################################################
#visualize the data

def visualize(file, x, y):
    #Reads the selected file, applies log transformation, and generates a scatter plot
    data = pd.read_csv(file) #read csv file

    #apply log10 transformation (replace 0s with NaN to avoid errors)
    data[x] = np.log10(data[x].replace(0, np.nan)) #transform x-axis data
    data[y] = np.log10(data[y].replace(0, np.nan)) #transform y-axis data
    data = data.dropna(subset=[x, y]) #drop any rows with NaN values

    data["Label"] = data["Cluster"].map(cluster_mapping) #map cluster numbers to labels

    #extract sample ID for title
    sample_id = extract_sample_id(file.name) #get sample ID from filename

    #separate noise and species
    noise = data[data["Cluster"] == -1] #filter noise
    species = data[data["Cluster"] != -1] #filter species

    #assign colours for species
    unique_clusters = sorted(species["Cluster"].unique()) #get unique species
    cmap = plt.get_cmap("turbo") #define colourmap
    cluster_colours = {cluster: cmap(i / max(1, len(unique_clusters) - 1)) for i, cluster in enumerate(unique_clusters)} #assign colours

    #plot the data
    plt.figure(figsize=(8, 6)) #define plot size

    #plot noise points
    if not noise.empty: #check if noise exists
        plt.scatter(noise[x], noise[y], c="gray", alpha=0.3, label="Noise", s=10) #plot noise

    #plot species
    for cluster in unique_clusters: #loop through species
        cluster_points = species[species["Cluster"] == cluster] #filter species data
        plt.scatter(
            cluster_points[x], cluster_points[y], #plot species
            color=to_rgba(cluster_colours[cluster]), label=cluster_mapping[cluster], alpha=0.8, s=15)

    #customize the plot
    plt.title(f"Species Visualization with noise excluded: {sample_id}") #set title
    plt.xlabel(f"log10({x})") #set x-axis label
    plt.ylabel(f"log10({y})") #set y-axis label
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Species") #add legend
    plt.tight_layout() #adjust layout
    plt.show() #display plot

################################################################
#main script

if __name__ == "__main__": #only execute if script is run directly
    print("cluster data visualization") #print introduction

    file = select_file() #prompt user to select a file
    if file: #if file is selected
        print(f"selected file: {file.name}") #print selected file
        data = pd.read_csv(file) #read file data
        x, y = select_params(data) #prompt user to select parameters
        visualize(file, x, y) #generate visualization
