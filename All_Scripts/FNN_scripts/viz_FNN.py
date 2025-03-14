#this script visualizes predicted data from the feedforward neural network (fnn)
#it allows users to select a prediction file, choose visualization parameters, and generate scatter plots

from pathlib import Path #for handling file operations
import pandas as pd #for data manipulation
import numpy as np #for numerical operations
import tensorflow as tf #for loading the trained fnn model
import joblib #for loading encoders and scalers
import matplotlib.pyplot as plt #for plotting
import random #for selecting random files
from sklearn.preprocessing import StandardScaler #for feature scaling
from matplotlib.colors import to_rgba #for colour mapping

################################################################
#set file paths  

test_folder=Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/split_data/Test_data") #test data folder
output_folder=Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/FNN_data/FNN_Predictions") #folder to save fnn predictions

model_path=Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/All_Scripts/fnn_scripts/custom_phyto_model.h5") #trained fnn model path
encoder_path=Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/All_Scripts/fnn_scripts/FNN_label_encoder.pkl") #label encoder path
scaler_path=Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/All_Scripts/fnn_scripts/FNN_scaler.pkl") #scaler path

CONFIDENCE_THRESHOLD=0.8 #set threshold for classification confidence

################################################################
#load model, encoder, and scaler  

model=tf.keras.models.load_model(model_path) #load trained fnn model
encoder=joblib.load(encoder_path) #load label encoder
scaler=joblib.load(scaler_path) #load feature scaler

print("\033[1;34m=======================\n      FNN Model Loaded       \n=======================\033[0m") #print success message
print("model path:",model_path) #print model path

output_folder.mkdir(parents=True,exist_ok=True) #ensure output directory exists

################################################################
#select file  

def select_file():
    #allows the user to select a prediction file manually or choose a random one
    files=[f.name for f in output_folder.glob("*_FNN.csv")] #get fnn prediction files
    if not files: #check if prediction files exist
        print("\033[1;31mno fnn prediction files found.\033[0m") #print warning
        return None 

    print("\navailable fnn prediction files:") #display options
    print("1. specify a file") #manual selection
    print("2. random file") #random selection
    choice=input("choose (1 or 2): ").strip()

    if choice=="2":
        return random.choice(files) #return random file
    elif choice=="1":
        for i,file in enumerate(files,1): #list available files
            print(f"{i}. {file}") 
        while True:
            try:
                index=int(input("file number: ").strip())-1 #get user input
                if 0<=index<len(files):
                    return files[index] #return selected file
                else:
                    print("invalid selection. try again.") #ask again
            except ValueError:
                print("invalid input. please enter a number.") #handle errors
    return None

################################################################
#select visualization parameters  

def select_params(data):
    #allows the user to choose predefined visualization parameters or manually select from available columns
    print("\nchoose visualization:") 
    print("1. FSC-H vs SSC-H") #first option
    print("2. FL3-H vs FL2-H") #second option
    print("3. custom") #manual selection
    choice=input("choose (1, 2, or 3): ").strip() 

    if choice=="1":
        return "FSC-H","SSC-H"
    elif choice=="2":
        return "FL3-H","FL2-H"
    
    cols=[col for col in data.columns if col not in ["Time","Label","Cluster"]] #exclude unnecessary columns
    print("\nselect x and y-axis parameters:") 
    for i,col in enumerate(cols,1): #list available columns
        print(f"{i}. {col}") 
    while True:
        try:
            x=cols[int(input("x-axis parameter: ").strip())-1] #get x-axis parameter
            y=cols[int(input("y-axis parameter: ").strip())-1] #get y-axis parameter
            return x,y 
        except (ValueError,IndexError):
            print("invalid selection. try again.") #ask again

################################################################
#visualize prediction data  

def visualize(file_path,x,y):
    #reads the selected file, applies log transformation, and generates a scatter plot
    data=pd.read_csv(file_path) #load file
    if {"Label",x,y}.issubset(data.columns): #check if required columns exist
        data[x]=np.log10(data[x].replace(0,np.nan)) #apply log transform to x
        data[y]=np.log10(data[y].replace(0,np.nan)) #apply log transform to y
        data=data.dropna(subset=[x,y]) #drop NaNs

        noise=data[data["Label"]=="Noise"] #extract noise points
        species=data[data["Label"]!="Noise"] #extract species points
        unique_species=sorted(species["Label"].unique()) #get unique species

        cmap=plt.get_cmap("turbo") #define colourmap
        species_colours={species:cmap(i/max(1,len(unique_species)-1)) for i,species in enumerate(unique_species)} #assign colours

        plt.figure(figsize=(8,6)) #set figure size

        #plot noise points
        if not noise.empty:
            plt.scatter(noise[x],noise[y],c="gray",alpha=0.3,label="Noise",s=10) #plot noise

        #plot species points
        for label in unique_species:
            species_points=species[species["Label"]==label] #filter species data
            plt.scatter(
                species_points[x],species_points[y], 
                color=to_rgba(species_colours[label]),label=label,alpha=0.8,s=15) #plot species

        #customize the plot
        plt.title(f"Predicted Species Visualization (FNN): {Path(file_path).name}") #set title
        plt.xlabel(f"log10({x})") #set x-axis label
        plt.ylabel(f"log10({y})") #set y-axis label
        plt.legend(bbox_to_anchor=(1.05,1),loc="upper left",title="Predicted Label") #add legend
        plt.tight_layout() #adjust layout
        plt.show() #display plot
    else:
        print(f"\033[1;31mskipping visualization: required columns missing in {file_path}\033[0m") #print error message

################################################################
#execute visualization  

if __name__=="__main__":
    print("\033[1;34m=======================\n  FNN Prediction Visualization  \n=======================\033[0m") #print script header
    file=select_file() #select prediction file

    if file: #if a file is selected
        file_path=output_folder/file #define full file path
        print(f"\n\033[1;36mselected file: {file}\033[0m") #print selected file
        data=pd.read_csv(file_path) #load file data
        x,y=select_params(data) #select visualization parameters
        visualize(file_path,x,y) #generate visualization
