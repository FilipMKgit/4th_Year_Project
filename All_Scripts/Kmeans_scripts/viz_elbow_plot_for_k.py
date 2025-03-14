#this script calculates the optimal number of clusters for k-means using an elbow plot
#it loads multiple CSV files, computes average values, and generates an elbow plot

from pathlib import Path #for handling file operations
import numpy as np #for numerical operations
import pandas as pd #for working with dataframes
import matplotlib.pyplot as plt #for plotting elbow plot
from sklearn.cluster import KMeans #for k-means clustering
from kneed import KneeLocator #for detecting the elbow point

################################################################
#define the directory where all CSV files are stored

data_dir = Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/working_preprocessed_data") #path to directory with processed data

################################################################
#function to load all CSV files in the directory and extract average values

def load_and_average_data(directory): #loads all CSV files and computes column-wise averages
    all_files = list(directory.glob("*.csv")) #find all CSV files
    avg_values_list = [] #initialize empty list to store averaged values
    
    for file_path in all_files: #loop through each CSV file
        print(f"loading file: {file_path.name}") #debugging print
        df = pd.read_csv(file_path) #load CSV file
        
        #select only numerical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns #get numerical columns

        #compute mean values for each numerical column
        avg_values = df[numeric_cols].mean().values #compute mean across rows

        #store only if numerical columns exist
        if avg_values.size > 0: #ensure valid data exists
            avg_values_list.append(avg_values) #add to list

    #convert list to a DataFrame for clustering
    if avg_values_list: #check if any valid data was extracted
        avg_df = pd.DataFrame(avg_values_list, columns=numeric_cols) #create dataframe with numerical column names
        print(f"loaded {len(avg_values_list)} files successfully!") #print total loaded files
        return avg_df #return dataframe with averaged values
    else:
        print("no valid numerical data found in any files!") #debugging message
        return None #return None if no valid data found

################################################################
#function to compute inertia values and generate elbow plot with elbow detection

def plot_elbow_method(data): #generates an elbow plot to find the optimal k
    if data is None or data.shape[0] < 2: #check if data is valid
        print("not enough data samples for clustering. more files are needed.") #print error message
        return
    
    #set range for k (ensuring it doesn't exceed the number of samples)
    k_values = range(1, min(10, data.shape[0]))  #ensure k doesn't exceed number of data points
    inertia = [] #initializes list to store inertia values

    #compute inertia for different k values
    for k in k_values: #loop over different cluster sizes
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10) #initializes k-means
        kmeans.fit(data) #fit model to data
        inertia.append(kmeans.inertia_) #store inertia value
    
    #detect the elbow point
    elbow_detector = KneeLocator(k_values, inertia, curve="convex", direction="decreasing") #initializes knee detector
    elbow_k = elbow_detector.elbow #find the optimal k
    
    #create the elbow plot with a professional, publication-style format
    plt.figure(figsize=(5, 4), dpi=150) #define plot size and resolution
    plt.plot(k_values, inertia, marker='o', color='black', linestyle='-', linewidth=1.5, markersize=6) #plot inertia values
    plt.xlabel("Number of Clusters (k)", fontsize=10, fontweight='bold') #set x-axis label
    plt.ylabel("Inertia (WCSS)", fontsize=10, fontweight='bold') #set y-axis label
    plt.title("Elbow Plot", fontsize=11, fontweight='bold') #set plot title

    #add a vertical dashed line at the detected elbow point
    if elbow_k: #check if an elbow point was detected
        plt.axvline(x=elbow_k, linestyle='--', color='red', linewidth=1.5, label=f"Optimal k = {elbow_k}") #draw vertical line
        plt.legend() #display legend
    
    #remove unnecessary grid lines
    plt.xticks(fontsize=9) #adjust x-ticks
    plt.yticks(fontsize=9) #adjust y-ticks
    plt.gca().spines["top"].set_visible(False) #hide top border
    plt.gca().spines["right"].set_visible(False) #hide right border
    #show the plot
    plt.show() #display elbow plot
    #print the optimal k value
    if elbow_k: #check if elbow point exists
        print(f"optimal number of clusters detected: k = {elbow_k}") #print optimal k value

################################################################
#run the full pipeline

if __name__ == "__main__": #ensure script runs as main
    print("loading and processing all files...") #print status
    avg_df = load_and_average_data(data_dir) #load and compute averages

    if avg_df is not None: #check if valid data was loaded
        print("generating elbow plot for clustering...") #print progress
        plot_elbow_method(avg_df) #run elbow plot function
    else:
        print("no valid data found in the directory. ensure csv files are present and contain numerical data.") #print error message
