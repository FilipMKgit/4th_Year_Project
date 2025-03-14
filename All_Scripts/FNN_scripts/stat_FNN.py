#this script evaluates cnn classification performance by comparing predicted labels  
#against ground truth labels from annotated cytometry data  

from pathlib import Path #for handling file operations
import numpy as np #for numerical operations
import pandas as pd #for data manipulation
import matplotlib.pyplot as plt #for plotting
import seaborn as sns #for heatmaps
import torch #for deep learning model handling
import torch.nn.functional as F #for model evaluation
from torch.utils.data import DataLoader, TensorDataset #for data batching
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import cv2 #for image processing in grad-cam
import torch.nn as nn #for defining model layers

################################################################
#define directories where different datasets are stored  

base_dir = Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data") #base directory for data storage
cnn_data_dir = Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/CNN_data/CNN_Predictions") #cnn prediction outputs
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
    return df[label_col].astype(int).values if label_col in df.columns else np.array([]) #return labels or empty array

################################################################
#compare cnn predictions with ground truth labels  

def evaluate_cnn(pred_files, gt_files):
    #computes accuracy, precision, recall, and confusion matrix for cnn classification
    scores = {"accuracy": [], "precision": [], "recall": []} #initialize score dictionary
    all_conf_matrices = [] #store confusion matrices
    all_labels = set() #collect all unique labels across datasets
    
    for preds, gt in zip(pred_files, gt_files): #iterate through files
        if "Cluster" not in preds.columns or "Cluster" not in gt.columns:
            continue #skip files missing cluster labels

        common_idx = preds.index.intersection(gt.index) #find common indices
        gt, preds = gt.loc[common_idx], preds.loc[common_idx] #filter to common rows

        true_labels = get_labels(gt, "Cluster") #extract ground truth labels
        pred_labels = get_labels(preds, "Cluster") #extract cnn predicted labels

        if true_labels.size == 0 or pred_labels.size == 0: #ensure both labels exist
            continue

        scores["accuracy"].append(accuracy_score(true_labels, pred_labels)) #compute accuracy
        scores["precision"].append(precision_score(true_labels, pred_labels, average='weighted', zero_division=0)) #compute precision
        scores["recall"].append(recall_score(true_labels, pred_labels, average='weighted', zero_division=0)) #compute recall

        #store all unique labels across datasets
        all_labels.update(np.unique(true_labels))

        #compute confusion matrix
        cm = confusion_matrix(true_labels, pred_labels, labels=np.unique(true_labels))
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

    return {k: np.mean(v) if v else None for k, v in scores.items()}, avg_conf_matrix #return evaluation metrics and confusion matrix

################################################################
#plot loss curve from training history  

def plot_loss_curve(history_file):
    #reads cnn training history and plots the loss curve
    df = pd.read_csv(history_file) #read training log
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss") #plot training loss
    plt.plot(df["epoch"], df["val_loss"], label="Validation Loss") #plot validation loss
    plt.xlabel("epoch") 
    plt.ylabel("loss") 
    plt.legend() 
    plt.title("cnn training loss curve") 
    plt.show() 

################################################################
#implement grad-cam to visualize important cnn features  

class GradCAM:
    #implements grad-cam for cnn visualization
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.hook = target_layer.register_backward_hook(self.save_gradient)

    def save_gradient(self, module, grad_input, grad_output):
        #saves gradients during backward pass
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx):
        #generates grad-cam heatmap for a specific class
        output = self.model(input_tensor)
        self.model.zero_grad()
        output[:, class_idx].backward()
        
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.target_layer(input_tensor).detach()

        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]
        
        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = torch.clamp(heatmap, min=0).numpy()
        heatmap = cv2.resize(heatmap, (input_tensor.shape[2], input_tensor.shape[3]))

        return heatmap

################################################################
#load and evaluate cnn predictions  

print("\nevaluating cnn - predictions vs annotated data...") 
metrics_cnn, avg_conf_matrix = evaluate_cnn(load_csv(cnn_data_dir), load_csv(ground_truth_dir)) #compare cnn predictions to ground truth

################################################################
#print performance results  

print("\n--- cnn model performance ---") 
print(f"accuracy: {metrics_cnn['accuracy']:.4f}" if metrics_cnn["accuracy"] is not None else "accuracy: n/a") 
print(f"precision: {metrics_cnn['precision']:.4f}" if metrics_cnn["precision"] is not None else "precision: n/a") 
print(f"recall: {metrics_cnn['recall']:.4f}" if metrics_cnn["recall"] is not None else "recall: n/a") 

################################################################
#plot average confusion matrix  

if avg_conf_matrix is not None:
    plt.figure(figsize=(8, 6)) 
    sns.heatmap(avg_conf_matrix, annot=True, fmt=".2f", cmap="Blues") 
    plt.xlabel("predicted labels") 
    plt.ylabel("true labels") 
    plt.title("average confusion matrix - cnn") 
    plt.show() 

################################################################
#load and plot loss curve (assuming a training log exists)  

history_file = base_dir / "cnn_training_history.csv" #training log file
if history_file.exists(): 
    plot_loss_curve(history_file) #plot loss curve
