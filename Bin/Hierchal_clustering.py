# this script applies hierarchical clustering to phototroph cytometry data  
# it randomly selects a CSV file, preprocesses the data, and generates a dendrogram  

from pathlib import Path  # modern file handling  
import random  # for selecting a random file  
import pandas as pd  # for handling dataframes  
import numpy as np  # for numerical operations  
from scipy.cluster.hierarchy import dendrogram, linkage  # for hierarchical clustering  
import matplotlib.pyplot as plt  # for visualization  
from sklearn.preprocessing import StandardScaler  # for normalizing data  

################################################################
# select a random CSV file  

def pick_csv(directory):
    # picks a random CSV file from the specified directory  
    csv_files = list(Path(directory).glob("*.csv"))  # find all CSV files  
    return random.choice(csv_files) if csv_files else None  # return a random file or None  

data_dir = Path(r"C:\Users\filip\OneDrive\Desktop\4YR_PROJ\raw_data\phototrophs")  # define path  
file = pick_csv(data_dir)  # select a random CSV file  

if file:
    random_data_selected = pd.read_csv(file)  # load csv  
else:
    raise FileNotFoundError(f"No CSV files found in {data_dir}")  # handle missing files  

################################################################
# preprocess the data  

workingd = random_data_selected[['FL3-H', 'FL2-H']]  # select relevant features  
datalog = np.log10(workingd + 1)  # apply log transformation  
datalog = datalog[datalog['FL2-H'] > 0.1]  # filter out low FL2-H values  
datalogscaled = StandardScaler().fit_transform(datalog)  # standardize data  

################################################################
# perform hierarchical clustering  

linkage_matrix = linkage(datalogscaled, method='ward')  # compute linkage matrix  

################################################################
# plot dendrogram  

plt.figure(figsize=(10, 7))  # set figure size  
dendrogram(linkage_matrix)  # generate dendrogram  
plt.title(f"Hierarchical Clustering\nFile: {file.name}")  # add title with filename  
plt.xlabel('Samples')  # x-axis label  
plt.ylabel('Distance')  # y-axis label  
plt.show()  # display plot  
