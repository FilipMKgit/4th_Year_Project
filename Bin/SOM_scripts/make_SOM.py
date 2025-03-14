# this script applies Self-Organizing Maps (SOM) to cluster cytometry data  
# it assigns each sample to a neuron in a 6x6 SOM grid and saves the results  

from pathlib import Path  # for handling file paths
import pandas as pd  # for working with dataframes
import numpy as np  # for numerical operations
import matplotlib.pyplot as plt  # for visualization
from minisom import MiniSom  # for Self-Organizing Maps (SOM)

################################################################
# define input and output directories  

input_dir = Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/working_data_standardized") # input folder
output_dir = Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/SOM_data/SOM") # output folder
output_dir.mkdir(parents=True, exist_ok=True)  # ensure output directory exists

grid_size = (6, 6)  # set SOM grid size (6x6 for 6 species)

################################################################
# process each file in the input directory  

for file_path in input_dir.glob("*.csv"):  # iterate over csv files
    print(f"processing {file_path.name}...")  # print current file being processed
    
    df = pd.read_csv(file_path)  # load csv into dataframe

    # select relevant features for SOM training
    features = ['FSC-H', 'SSC-H', 'FL2-H', 'FL3-H']  # feature columns
    data = df[features].values  # extract numerical data

    ################################################################
    # apply SOM  

    som = MiniSom(x=grid_size[0], y=grid_size[1], input_len=data.shape[1], sigma=1.0, learning_rate=0.5)  # initialize SOM
    som.random_weights_init(data)  # initialize weights randomly
    som.train_random(data, num_iteration=10000)  # train SOM on data

    # assign each sample to a SOM neuron
    assigned_clusters = np.array([som.winner(d) for d in data])  # get neuron coordinates
    df["SOM_Cluster"] = [f"{x}-{y}" for x, y in assigned_clusters]  # store cluster labels as string

    # fix potential Excel misinterpretation by forcing text format
    df["SOM_Cluster"] = "'" + df["SOM_Cluster"].astype(str)  # add leading apostrophe

    # create a new numeric cluster column
    df["Cluster"] = [x * grid_size[1] + y for x, y in assigned_clusters]  # convert (x, y) to single numeric label

    ################################################################
    # save results  

    output_file_path = output_dir / f"{file_path.stem}_SOM.csv"  # construct output filename
    df.to_csv(output_file_path, index=False)  # save dataframe as CSV
    print(f"som clustering done for '{file_path.name}'. saved to: {output_file_path}")  # print status

################################################################
# completion message  

print("\033[1;32mall som clustering tasks finished! files are saved.\033[0m")  # print success message  
