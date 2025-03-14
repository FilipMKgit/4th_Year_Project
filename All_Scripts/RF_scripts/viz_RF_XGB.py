#this script visualizes classification results from random forest (rf) and xgboost (xgb)
#it allows users to select a prediction file, choose visualization parameters, and generate scatter plots

from pathlib import Path #for handling file operations
import pandas as pd #for data manipulation
import numpy as np #for numerical operations
import random #for selecting random files
import matplotlib.pyplot as plt #for plotting

################################################################
#set directories  

rf_output_dir = Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/RandomForest_data/RF") #random forest results
xgb_output_dir = Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/RandomForest_data/XGBoost") #xgboost results

################################################################
#select dataset type  

def get_files(folder, pattern):
    #retrieves list of available prediction files matching a pattern
    return [f.name for f in folder.glob(pattern)] #get matching files

def pick_file():
    #allows user to select a prediction file from rf or xgb results
    print("\nselect dataset type:") 
    print("1. random forest") #first option
    print("2. xgboost") #second option
    
    choice = input("enter number (1-2): ").strip() 

    if choice == "1":
        folder = rf_output_dir #set rf directory
        file_pattern = "*_RF_predicted.csv" #rf filename pattern
        dataset_type = "rf" 
    elif choice == "2":
        folder = xgb_output_dir #set xgb directory
        file_pattern = "*_XGBoost_predicted.csv" #xgb filename pattern
        dataset_type = "xgb"
    else:
        print("invalid selection. defaulting to rf.") #handle invalid input
        folder = rf_output_dir #set default directory
        file_pattern = "*_RF_predicted.csv" 
        dataset_type = "rf"

    files = get_files(folder, file_pattern) #get matching files
    
    if not files: #check if files exist
        print(f"no {dataset_type} files found.") #print warning
        return None, None 

    print("\n1. choose a specific file") #manual selection
    print("2. pick a random file") #random selection
    
    user_choice = input("enter 1 or 2: ").strip() 

    if user_choice == "2":
        return folder, random.choice(files) #return random file

    if user_choice == "1":
        for i, file in enumerate(files, 1): #list available files
            print(f"{i}. {file}") 
        while True:
            selection = input("enter file number: ").strip() 
            if selection.isdigit():
                idx = int(selection) - 1 
                if 0 <= idx < len(files):
                    return folder, files[idx] #return selected file
    
    return folder, files[0] #default return first file

################################################################
#select plot parameters  

def pick_plot_columns(df):
    #allows user to choose x and y-axis parameters for visualization
    print("\nchoose plot parameters:") 
    print("1. FSC-G vs SSC-H") #first option
    print("2. FL3-H vs FL2-H") #second option
    print("3. custom selection") #manual selection
    
    user_input = input("enter number: ").strip() 

    if user_input == "1":
        return "FSC-H", "SSC-H" 
    elif user_input == "2":
        return "FL3-H", "FL2-H"
    elif user_input == "3":
        cols = [col for col in df.columns if col not in ["Time", "Predicted_Label", "Cluster"]] #get valid columns
        print("\nselect x and y-axis parameters:") 
        for i, col in enumerate(cols, 1): #list available columns
            print(f"{i}. {col}") 
        try:
            x_idx = int(input("x-axis column number: ")) - 1 #get x-axis parameter
            y_idx = int(input("y-axis column number: ")) - 1 #get y-axis parameter
            return cols[x_idx], cols[y_idx] #return chosen parameters
        except (ValueError, IndexError):
            return "FSC-H", "SSC-H" 

    return "FSC-H", "SSC-H" 

################################################################
#visualize classification output  

def plot_clusters(df, file_path, x, y):
    #plots classification results from rf or xgb, mapping clusters to predicted labels using the turbo colourmap
    
    if "Cluster" not in df.columns or "Predicted_Label" not in df.columns: #check required columns
        print(f"skipping {file_path.name}: missing 'cluster' or 'predicted_label' column.") #print warning
        return

    df[x] = np.log10(df[x].replace(0, np.nan)) #apply log transform to x
    df[y] = np.log10(df[y].replace(0, np.nan)) #apply log transform to y
    df.dropna(subset=[x, y], inplace=True) #remove NaNs

    cluster_to_label = df.groupby("Cluster")["Predicted_Label"].first().to_dict() #map cluster id to predicted labels

    unique_clusters = sorted(df["Cluster"].unique()) #get unique clusters
    turbo_cmap = plt.get_cmap("turbo", len(unique_clusters)) #define colourmap
    cluster_colours = {cluster: turbo_cmap(i / len(unique_clusters)) for i, cluster in enumerate(unique_clusters)} #assign colours

    plt.figure(figsize=(8, 6)) #set figure size
    for cluster in unique_clusters: #plot each cluster
        subset_df = df[df["Cluster"] == cluster] #filter species data
        label = cluster_to_label.get(cluster, f"Cluster {cluster}") #default label if missing
        plt.scatter(
            subset_df[x], subset_df[y],
            color=cluster_colours[cluster], label=label, alpha=0.7, s=15
        ) #plot species

    plt.xlabel(f"log10({x})") #set x-axis label
    plt.ylabel(f"log10({y})") #set y-axis label
    plt.title(f"{Path(file_path).name} - Cluster Visualization") #set title
    plt.legend(title="Predicted Labels", bbox_to_anchor=(1.05, 1), loc='upper left') #add legend
    plt.tight_layout() #adjust layout
    plt.show() #display plot

################################################################
#execute visualization  

if __name__ == "__main__":
    print("\033[1;34m=======================\n  Visualization for RF and XGB Results  \n=======================\033[0m") #print script header

    folder, file_name = pick_file() #select file
    
    if file_name: #if a file is selected
        file_path = folder / file_name #define full file path
        df_data = pd.read_csv(file_path) #load file data
        x_col, y_col = pick_plot_columns(df_data) #select visualization parameters
        plot_clusters(df_data, file_path, x_col, y_col) #generate visualization
