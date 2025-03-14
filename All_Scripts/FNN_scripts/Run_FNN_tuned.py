#this script applies Feedforward Neural Network (FNN) for classification
#it loads a trained model, applies predictions to test data, and assigns labels based on confidence thresholds

from pathlib import Path #import modern file handling
import pandas as pd #import dataframe operations
import numpy as np #import numerical operations
import tensorflow as tf #import deep learning framework
import joblib #import for saving/loading models
from sklearn.preprocessing import StandardScaler #import scaler for normalization
from collections import Counter #import counter for class balancing

################################################################
#file paths  

test_folder=Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/split_data/Test_data") #test data folder
output_folder=Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/FNN_data/FNN_Predictions") #output folder for FNN predictions

model_path=Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/All_Scripts/fnn_scripts/custom_phyto_model.h5") #FNN model save path
encoder_path=Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/All_Scripts/fnn_scripts/FNN_label_encoder.pkl") #label encoder save path
scaler_path=Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/All_Scripts/fnn_scripts/FNN_scaler.pkl") #scaler save path

CONFIDENCE_THRESHOLD=0.8 #set threshold for classification confidence

################################################################
#load model, encoder, and scaler  

model=tf.keras.models.load_model(model_path) #load trained model
encoder=joblib.load(encoder_path) #load label encoder
scaler=joblib.load(scaler_path) #load feature scaler

print("\033[1;34m=======================\n      FNN Model Loaded       \n=======================\033[0m") #print model loaded message
print("Model path:",model_path) #print model path

output_folder.mkdir(parents=True,exist_ok=True) #ensure output directory exists

################################################################
#cluster mapping  

cluster_mapping={"Crypto":1,"Nano1":2,"Nano2":3,"PEuk1":4,"PEuk2":5,"Syn":6,"Noise":-1} #define mapping

test_files=[f for f in test_folder.iterdir() if f.suffix==".csv"] #get test csv files

################################################################
#process each test file  

for test_file in test_files: #loop through test files
    print(f"\n\033[1;36mProcessing {test_file.name}...\033[0m") #print status message
    df_test=pd.read_csv(test_file) #load test file

    features=["FSC-H","SSC-H","FL2-H","FL3-H"] #define required features
    missing_cols=[col for col in features if col not in df_test.columns] #check for missing columns
    if missing_cols: #if columns are missing, skip file
        print(f"\033[1;33mSkipping {test_file.name} - Missing columns: {missing_cols}\033[0m") #print warning
        continue #skip file

    X_test=df_test[features].values #extract feature values
    X_test=scaler.transform(X_test) #scale test data

    predictions=model.predict(X_test) #make predictions
    max_probs=np.max(predictions,axis=1) #get highest probability for each sample
    predicted_classes=np.argmax(predictions,axis=1) #get predicted class indices
    predicted_labels=encoder.inverse_transform(predicted_classes) #decode class labels

    #apply confidence threshold
    final_labels=["Noise" if prob<CONFIDENCE_THRESHOLD else label for prob,label in zip(max_probs,predicted_labels)]
    df_test["Label"] = final_labels #assign final labels

    if "Label" not in df_test.columns: #check if label column exists
        print(f"\033[1;33mSkipping {test_file.name} - 'Label' column missing\033[0m") #print warning
        continue #skip this file

    # Ensure label exists in cluster mapping before assigning cluster
    df_test["Cluster"] = df_test["Label"].apply(lambda x: cluster_mapping.get(x, -1)) #assign cluster numbers safely

    #generate output filename
    new_file_name=test_file.stem+"_FNN"+test_file.suffix #modify filename
    output_path=output_folder/new_file_name #define output path
    
    df_test.to_csv(output_path,index=False) #save predictions
    print(f"\033[1;32mPredictions saved:\033[0m {output_path}") #print success message

################################################################
#print completion message  

print("\n\033[1;32m=======================\n      FNN Predictions Completed       \n=======================\033[0m") #print success message
print("Predictions saved in:",output_folder) #print output directory
