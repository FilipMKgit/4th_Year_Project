#this script evaluates lda classification performance by comparing predicted labels  
#against ground truth labels from annotated cytometry data  

from pathlib import Path #for handling file operations
import numpy as np #for numerical operations
import pandas as pd #for data manipulation
import matplotlib.pyplot as plt #for plotting confusion matrix
import seaborn as sns #for heatmaps
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, f1_score, confusion_matrix
from scipy.optimize import linear_sum_assignment #for optimal cluster label alignment

################################################################
#define directories where different datasets are stored  

base_dir = Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data") #base directory for data storage
lda_data_dir = base_dir / "LDA_data" #lda prediction outputs
ground_truth_dir = base_dir / "annotated_data_with_noise" #annotated ground truth data

################################################################
#function to load all csv files from a directory  

def load_csv(path):
    #loads all csv files from the given directory and returns a list of dataframes
    all_files = [f for f in path.glob("*.csv")] #find all csv files
    return [pd.read_csv(f) for f in all_files] #read all csvs

################################################################
#function to extract labels from dataframe  

def get_labels(df, label_col):
    #extracts class labels from the dataframe column
    return df[label_col].astype(str).values if label_col in df.columns else np.array([]) #return labels or empty array

################################################################
#align predicted cluster labels with true labels  

def align_clusters(true_labels, pred_labels):
    #reorders predicted labels to best match true labels using optimal assignment
    unique_t, unique_p = np.unique(true_labels), np.unique(pred_labels) #get unique cluster ids
    cost = np.zeros((len(unique_t), len(unique_p))) #initialize cost matrix

    for i, t in enumerate(unique_t): #loop through true labels
        for j, p in enumerate(unique_p): #loop through predicted labels
            cost[i, j] = -np.sum((true_labels == t) & (pred_labels == p)) #compute cost

    row_ind, col_ind = linear_sum_assignment(cost) #solve optimal assignment
    mapping = {unique_p[j]: unique_t[i] for i, j in zip(row_ind, col_ind)} #create mapping

    return np.array([mapping.get(label, -1) for label in pred_labels]) #apply mapping

################################################################
#function to compare lda clustering results with ground truth  

def compare_to_gt(cluster_files, gt_files):
    #computes ari, nmi, f1, and confusion matrix for clustering performance evaluation
    scores = {"ari": [], "nmi": [], "f1": []} #initialize score dictionary
    all_conf_matrices = [] #store confusion matrices
    all_labels = set() #store all unique labels across datasets
    
    for clusters, gt in zip(cluster_files, gt_files): #iterate through files
        if "Label" not in clusters.columns or "Cluster" not in gt.columns:
            continue #skip files missing required labels

        common_idx = clusters.index.intersection(gt.index) #find common indices
        gt, clusters = gt.loc[common_idx], clusters.loc[common_idx] #filter to common rows

        true_labels = get_labels(gt, "Cluster") #extract ground truth labels
        pred_labels = get_labels(clusters, "Label") #extract lda predicted labels

        if true_labels.size == 0 or pred_labels.size == 0: #ensure both labels exist
            continue

        #convert categorical lda labels to numeric
        unique_pred_classes = {name: idx for idx, name in enumerate(np.unique(pred_labels))}
        pred_labels_numeric = np.array([unique_pred_classes[label] for label in pred_labels])

        #align predicted labels to true clusters
        pred_labels_aligned = align_clusters(true_labels, pred_labels_numeric)

        scores["ari"].append(adjusted_rand_score(true_labels, pred_labels_aligned)) #compute ari
        scores["nmi"].append(normalized_mutual_info_score(true_labels, pred_labels_aligned)) #compute nmi
        scores["f1"].append(f1_score(true_labels, pred_labels_aligned, average='weighted')) #compute f1-score

        #store all unique labels across datasets
        all_labels.update(np.unique(true_labels))

        #compute confusion matrix
        cm = confusion_matrix(true_labels, pred_labels_aligned, labels=np.unique(true_labels))
        all_conf_matrices.append(cm)

    #standardize confusion matrix sizes
    all_labels = sorted(all_labels) #ensure consistent label order
    num_labels = len(all_labels)
    standardized_matrices = []

    for cm in all_conf_matrices:
        if cm.shape != (num_labels, num_labels):
            new_cm = np.zeros((num_labels, num_labels)) #create empty matrix
            min_dim = min(cm.shape[0], num_labels)
            new_cm[:min_dim, :min_dim] = cm[:min_dim, :min_dim] #fill in available values
            standardized_matrices.append(new_cm)
        else:
            standardized_matrices.append(cm)

    avg_conf_matrix = np.mean(standardized_matrices, axis=0) if standardized_matrices else None #compute average confusion matrix

    return {k: np.mean(v) if v else None for k, v in scores.items()}, avg_conf_matrix, all_labels #return evaluation metrics, confusion matrix, and labels

################################################################
#comparing lda clustering results with annotated data  

print("\nevaluating lda - data vs annotated data...") 
metrics_lda, avg_conf_matrix, unique_labels = compare_to_gt(load_csv(lda_data_dir), load_csv(ground_truth_dir)) #compare lda results to ground truth

################################################################
#print performance results  

print("\n--- lda clustering performance ---") 
print(f"ari: {metrics_lda['ari']:.4f}") if metrics_lda["ari"] is not None else print("ari: n/a") 
print(f"nmi: {metrics_lda['nmi']:.4f}") if metrics_lda["nmi"] is not None else print("nmi: n/a") 
print(f"f1: {metrics_lda['f1']:.4f}") if metrics_lda["f1"] is not None else print("f1: n/a") 

################################################################
#plot average confusion matrix  

if avg_conf_matrix is not None:
    plt.figure(figsize=(8, 6)) 
    sns.heatmap(avg_conf_matrix, annot=True, fmt=".2f", cmap="Blues", 
                xticklabels=unique_labels, yticklabels=unique_labels) 
    plt.xlabel("predicted labels") 
    plt.ylabel("true labels") 
    plt.title("average confusion matrix - lda clustering") 
    plt.show() 
