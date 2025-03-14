#this script evaluates clustering performance by comparing GMM cluster assignments  
#against ground truth labels from annotated cytometry data  

from pathlib import Path #for handling file operations
import numpy as np #for numerical operations
import pandas as pd #for data manipulation
from sklearn.mixture import GaussianMixture #for GMM clustering
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score, f1_score
from scipy.optimize import linear_sum_assignment #for optimal cluster label alignment

################################################################
#set directories where different datasets are stored  

base_dir = Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data") #base path for all data
original_data_dir = base_dir / "working_preprocessed_data" #original preprocessed data
pca_data_dir = base_dir / "PCA_data" #pca-transformed data
tsne_data_dir = base_dir / "tSNE_data" #tsne-transformed data
umap_data_dir = base_dir / "UMAP_data" #umap-transformed data

gmm_original_dir = base_dir / "GMM_data/gmm_data" #gmm results on original data
gmm_pca_dir = base_dir / "GMM_data/gmm_data_PCA" #gmm results on pca data
gmm_tsne_dir = base_dir / "GMM_data/gmm_data_tSNE" #gmm results on tsne data
gmm_umap_dir = base_dir / "GMM_data/gmm_data_UMAP" #gmm results on umap data

ground_truth_dir = base_dir / "annotated_data_with_noise" #ground truth data with noise

#################################################################
#function to load all csv files from a directory  

def load_csv(path):
    #attempts to load all csv files from the given directory
    try:
        return [pd.read_csv(f) for f in path.glob("*.csv")] #read all csv files
    except Exception:
        return [] #return empty list if no files found

################################################################
#extracts cluster labels from a dataframe  

def get_labels(df):
    #loop through possible column names
    for col in ["Cluster", "cluster"]:
        if col in df.columns:
            return df[col].values #return cluster labels as numpy array
    return np.array([]) #return empty array if no cluster column is found

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
#function to compare clustering results with ground truth  

def compare_to_gt(cluster_files, gt_files):
    #computes ari, nmi, f1, and silhouette score for clustering performance evaluation
    scores = {"ari": [], "nmi": [], "f1": [], "silhouette": []} #initialize score dictionary

    for clusters, gt in zip(cluster_files, gt_files): #iterate through files
        if "FL2-H" not in gt.columns: #ensure relevant feature is present
            continue

        common_idx = clusters.index.intersection(gt.index) #find common indices
        gt, clusters = gt.loc[common_idx], clusters.loc[common_idx] #filter to common rows
        
        true_labels, pred_labels = get_labels(gt), get_labels(clusters) #extract cluster labels
        if true_labels.size == 0 or pred_labels.size == 0: #ensure both labels exist
            continue

        pred_labels = align_clusters(true_labels, pred_labels) #align cluster labels

        scores["ari"].append(adjusted_rand_score(true_labels, pred_labels)) #compute ari
        scores["nmi"].append(normalized_mutual_info_score(true_labels, pred_labels)) #compute nmi
        scores["f1"].append(f1_score(true_labels, pred_labels, average='weighted')) #compute f1-score

        #compute silhouette score only if there are at least 2 clusters
        unique_clusters = np.unique(pred_labels) #get unique clusters
        if len(unique_clusters) > 1:
            try:
                scores["silhouette"].append(silhouette_score(gt.iloc[:, :-1], pred_labels)) #compute silhouette score
            except:
                scores["silhouette"].append(None) #handle exceptions
        else:
            scores["silhouette"].append(None) #assign none if only one cluster

    return {k: np.mean(v) if v else None for k, v in scores.items()} #return average scores

################################################################
#evaluating gmm clustering results against annotated ground truth  

print("\nevaluating gmm - original data vs annotated data...")
metrics_gt_orig = compare_to_gt(load_csv(gmm_original_dir), load_csv(ground_truth_dir)) #evaluate gmm on original data

print("\nevaluating gmm - pca data vs annotated data...")
metrics_gt_pca = compare_to_gt(load_csv(gmm_pca_dir), load_csv(ground_truth_dir)) #evaluate gmm on pca data

print("\nevaluating gmm - tsne data vs annotated data...")
metrics_gt_tsne = compare_to_gt(load_csv(gmm_tsne_dir), load_csv(ground_truth_dir)) #evaluate gmm on tsne data

print("\nevaluating gmm - umap data vs annotated data...")
metrics_gt_umap = compare_to_gt(load_csv(gmm_umap_dir), load_csv(ground_truth_dir)) #evaluate gmm on umap data

################################################################
#print performance results  

print("\n--- clustering performance ---") 
print(f"original data - ari: {metrics_gt_orig['ari']:.4f}, nmi: {metrics_gt_orig['nmi']:.4f}, f1: {metrics_gt_orig['f1']:.4f}, silhouette: {metrics_gt_orig['silhouette']:.4f}") 
print(f"pca data - ari: {metrics_gt_pca['ari']:.4f}, nmi: {metrics_gt_pca['nmi']:.4f}, f1: {metrics_gt_pca['f1']:.4f}, silhouette: {metrics_gt_pca['silhouette']:.4f}") 
print(f"tsne data - ari: {metrics_gt_tsne['ari']:.4f}, nmi: {metrics_gt_tsne['nmi']:.4f}, f1: {metrics_gt_tsne['f1']:.4f}, silhouette: {metrics_gt_tsne['silhouette']:.4f}") 
print(f"umap data - ari: {metrics_gt_umap['ari']:.4f}, nmi: {metrics_gt_umap['nmi']:.4f}, f1: {metrics_gt_umap['f1']:.4f}, silhouette: {metrics_gt_umap['silhouette']:.4f}") 
