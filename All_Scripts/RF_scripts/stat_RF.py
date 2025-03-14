#this script evaluates random forest classification performance by comparing predicted labels  
#against ground truth labels from annotated cytometry data  

from pathlib import Path #for handling file operations
import numpy as np #for numerical operations
import pandas as pd #for data manipulation
import matplotlib.pyplot as plt #for plotting confusion matrix
import seaborn as sns #for heatmaps
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

################################################################
#define directories where different datasets are stored  

base_dir = Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data") #base directory for data storage
rf_data_dir = base_dir / "RandomForest_data" / "RF" #rf prediction outputs
ground_truth_dir = base_dir / "split_data" / "Test_data_annotated" #annotated ground truth data

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
#function to evaluate rf predictions vs ground truth  

def evaluate_rf(pred_files, gt_files):
    #computes accuracy, precision, recall, and confusion matrix for rf predictions
    scores = {"accuracy": [], "precision": [], "recall": []} #initialize score dictionary
    all_conf_matrices = [] #store confusion matrices
    all_labels = set() #store all unique labels across datasets
    
    for preds, gt in zip(pred_files, gt_files): #iterate through files
        if "Predicted_Label" not in preds.columns or "Cluster" not in gt.columns or "Label" not in gt.columns:
            continue #skip files missing required labels

        #map predicted_label to cluster using label column  
        label_to_cluster = gt.set_index("Label")["Cluster"].to_dict() #create mapping from label to cluster
        preds["Predicted_Cluster"] = preds["Predicted_Label"].map(label_to_cluster) #assign cluster labels

        #align datasets based on common features  
        common_columns = ["FSC-H", "SSC-H", "Time"] #shared feature columns
        merged_df = gt.merge(preds, on=common_columns, suffixes=("_true", "_pred")) #merge ground truth and predictions
        
        true_labels = merged_df["Cluster"].values #extract true labels
        pred_labels = merged_df["Predicted_Cluster"].values #extract predicted labels

        #ensure all unique labels are considered, including -1  
        unique_labels = np.unique(np.concatenate((true_labels, pred_labels, [-1]))) 
        
        scores["accuracy"].append(accuracy_score(true_labels, pred_labels)) #compute accuracy
        scores["precision"].append(precision_score(true_labels, pred_labels, average='weighted', zero_division=0, labels=unique_labels)) #compute precision
        scores["recall"].append(recall_score(true_labels, pred_labels, average='weighted', zero_division=0, labels=unique_labels)) #compute recall
        
        #store all unique labels across datasets  
        all_labels.update(unique_labels) 

        #compute confusion matrix  
        cm = confusion_matrix(true_labels, pred_labels, labels=unique_labels) 
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
#evaluate rf predictions  

print("\nevaluating random forest - predictions vs annotated data...") 
metrics_rf, avg_conf_matrix, unique_labels = evaluate_rf(load_csv(rf_data_dir), load_csv(ground_truth_dir)) #compare rf results to ground truth

################################################################
#print performance results  

print("\n--- random forest model performance ---") 
print(f"accuracy: {metrics_rf['accuracy']:.4f}" if metrics_rf["accuracy"] is not None else "accuracy: n/a") 
print(f"precision: {metrics_rf['precision']:.4f}" if metrics_rf["precision"] is not None else "precision: n/a") 
print(f"recall: {metrics_rf['recall']:.4f}" if metrics_rf["recall"] is not None else "recall: n/a") 

################################################################
#plot average confusion matrix  

if avg_conf_matrix is not None:
    plt.figure(figsize=(8, 6)) 
    sns.heatmap(avg_conf_matrix, annot=True, fmt=".2f", cmap="Blues", 
                xticklabels=unique_labels, yticklabels=unique_labels) 
    plt.xlabel("predicted labels") 
    plt.ylabel("true labels") 
    plt.title("average confusion matrix - random forest") 
    plt.show() 
