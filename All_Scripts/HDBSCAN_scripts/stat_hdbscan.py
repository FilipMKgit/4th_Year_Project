from pathlib import Path #for handling file operations
import numpy as np #for numerical operations
import pandas as pd #for data manipulation
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score, f1_score
from scipy.optimize import linear_sum_assignment #for aligning predicted cluster labels with ground truth

################################################################
#set directories where different datasets are stored  

base_dir = Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data") #base path for data storage
original_data_dir = base_dir / "working_preprocessed_data" #original untransformed data
pca_data_dir = base_dir / "PCA_data" #PCA-transformed data
tsne_data_dir = base_dir / "tSNE_data" #t-SNE-transformed data
umap_data_dir = base_dir / "UMAP_data" #UMAP-transformed data

hdbscan_original_dir = base_dir / "HDBSCAN_data/hdbscan_data" #HDBSCAN on original data
hdbscan_pca_dir = base_dir / "HDBSCAN_data/hdbscan_data_PCA" #HDBSCAN on PCA data
hdbscan_tsne_dir = base_dir / "HDBSCAN_data/hdbscan_data_tSNE" #HDBSCAN on t-SNE data
hdbscan_umap_dir = base_dir / "HDBSCAN_data/hdbscan_data_UMAP" #HDBSCAN on UMAP data

ground_truth_dir = base_dir / "annotated_data_with_noise" #annotated ground truth data

################################################################
#function to load all CSV files from a directory  

def load_csv(path):
    #attempt to load all CSV files from the given directory
    try:
        return [pd.read_csv(f) for f in path.glob("*.csv")] #return list of dataframes
    except Exception:
        return [] #return empty list if loading fails

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
    unique_t, unique_p = np.unique(true_labels), np.unique(pred_labels) #find unique labels
    cost = np.zeros((len(unique_t), len(unique_p))) #initialize cost matrix

    #calculate cost based on label matches
    for i, t in enumerate(unique_t):
        for j, p in enumerate(unique_p):
            cost[i, j] = -np.sum((true_labels == t) & (pred_labels == p))

    row_ind, col_ind = linear_sum_assignment(cost) #find optimal label assignment
    mapping = {unique_p[j]: unique_t[i] for i, j in zip(row_ind, col_ind)} #map predicted labels to true labels
    return np.array([mapping.get(label, -1) for label in pred_labels]) #return aligned labels

################################################################
#compares clustering results with ground truth  

def compare_to_gt(cluster_files, gt_files):
    scores = {"ari": [], "nmi": [], "f1": [], "silhouette": []} #initialize score storage

    #loop through each file pair
    for clusters, gt in zip(cluster_files, gt_files):
        if "FL2-H" not in gt.columns: #ensure valid file
            continue

        #find common indices between cluster file and ground truth
        common_idx = clusters.index.intersection(gt.index)
        gt, clusters = gt.loc[common_idx], clusters.loc[common_idx]

        true_labels, pred_labels = get_labels(gt), get_labels(clusters) #extract labels
        if true_labels.size == 0 or pred_labels.size == 0: #check for empty labels
            continue

        pred_labels = align_clusters(true_labels, pred_labels) #align cluster labels

        scores["ari"].append(adjusted_rand_score(true_labels, pred_labels)) #ARI score
        scores["nmi"].append(normalized_mutual_info_score(true_labels, pred_labels)) #NMI score
        scores["f1"].append(f1_score(true_labels, pred_labels, average='weighted')) #F1 score

        #compute silhouette score only if more than one cluster exists
        unique_clusters = np.unique(pred_labels)
        if len(unique_clusters) > 1:
            try:
                silhouette = silhouette_score(gt.iloc[:, :-1], pred_labels)
                scores["silhouette"].append(silhouette)
            except:
                pass #ignore silhouette score errors

    #compute mean of scores while skipping None values
    return {k: np.mean([x for x in v if x is not None]) if any(v) else None for k, v in scores.items()}

################################################################
#compare clustering results with annotated data  

print("\nEvaluating HDBSCAN - Original Data vs Annotated data...")
metrics_gt_orig = compare_to_gt(load_csv(hdbscan_original_dir), load_csv(ground_truth_dir))

print("\nEvaluating HDBSCAN - PCA Data vs Annotated data...")
metrics_gt_pca = compare_to_gt(load_csv(hdbscan_pca_dir), load_csv(ground_truth_dir))

print("\nEvaluating HDBSCAN - tSNE Data vs Annotated data...")
metrics_gt_tsne = compare_to_gt(load_csv(hdbscan_tsne_dir), load_csv(ground_truth_dir))

print("\nEvaluating HDBSCAN - UMAP Data vs Annotated data...")
metrics_gt_umap = compare_to_gt(load_csv(hdbscan_umap_dir), load_csv(ground_truth_dir))

################################################################
#print performance results  

print("\n--- Clustering Performance ---")
print(f"Original Data - ARI: {metrics_gt_orig['ari']:.4f}, NMI: {metrics_gt_orig['nmi']:.4f}, F1: {metrics_gt_orig['f1']:.4f}, Silhouette: {metrics_gt_orig['silhouette']:.4f}")
print(f"PCA Data - ARI: {metrics_gt_pca['ari']:.4f}, NMI: {metrics_gt_pca['nmi']:.4f}, F1: {metrics_gt_pca['f1']:.4f}, Silhouette: {metrics_gt_pca['silhouette']:.4f}")
print(f"tSNE Data - ARI: {metrics_gt_tsne['ari']:.4f}, NMI: {metrics_gt_tsne['nmi']:.4f}, F1: {metrics_gt_tsne['f1']:.4f}, Silhouette: {metrics_gt_tsne['silhouette']:.4f}")
print(f"UMAP Data - ARI: {metrics_gt_umap['ari']:.4f}, NMI: {metrics_gt_umap['nmi']:.4f}, F1: {metrics_gt_umap['f1']:.4f}, Silhouette: {metrics_gt_umap['silhouette']:.4f}")
