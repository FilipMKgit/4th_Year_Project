# this script applies Self-Organizing Maps (SOM) for dimensionality reduction  
# and then clusters the results using HDBSCAN  

from pathlib import Path  # for handling file operations
import pandas as pd  # for data manipulation
import numpy as np  # for numerical operations
import matplotlib.pyplot as plt  # for plotting
from minisom import MiniSom  # for Self-Organizing Maps (SOM)
import hdbscan  # for clustering

################################################################
# define input and output directories  

input_dir = Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/working_data_standardized") # input folder
output_dir = Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/SOM_data/HDBSCAN_SOM") # output folder
output_dir.mkdir(parents=True, exist_ok=True)  # ensure output directory exists

grid_size = (6, 6)  # define SOM grid size
min_cluster_size = 50  # minimum number of samples for HDBSCAN clusters

################################################################
# process each file in the input directory  

for file_path in input_dir.glob("*.csv"): # iterate over all CSV files
    print(f"processing {file_path.name}...") # print current file being processed
    
    df = pd.read_csv(file_path) # load csv into dataframe

    # select relevant features for SOM training
    features = ['FSC-H', 'SSC-H', 'FL2-H', 'FL3-H'] # feature columns
    data = df[features].values # extract numerical data

    ################################################################
    # apply SOM  

    som = MiniSom(x=grid_size[0], y=grid_size[1], input_len=data.shape[1], sigma=1.0, learning_rate=0.5) # initialize SOM
    som.random_weights_init(data) # initialize weights
    som.train_random(data, num_iteration=10000) # train SOM with data

    # assign each sample to a SOM neuron
    assigned_clusters = np.array([som.winner(d) for d in data]) # get neuron coordinates
    df["SOM_Cluster"] = [f"{x}-{y}" for x, y in assigned_clusters] # store cluster labels as string
    
    # convert SOM cluster labels to numeric coordinates
    df["SOM_X"] = [x for x, y in assigned_clusters] # extract x-coordinates
    df["SOM_Y"] = [y for x, y in assigned_clusters] # extract y-coordinates
    
    ################################################################
    # apply HDBSCAN  

    hdb = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, gen_min_span_tree=True) # initialize HDBSCAN
    df["HDBSCAN_Cluster"] = hdb.fit_predict(df[["SOM_X", "SOM_Y"]]) # cluster based on SOM positions

    ################################################################
    # save results  

    output_file_path = output_dir / f"{file_path.stem}_HDBSCAN_SOM.csv" # construct output filename
    df.to_csv(output_file_path, index=False) # save dataframe as CSV
    print(f"som + hdbscan clustering done for '{file_path.name}'. saved to: {output_file_path}") # print status
