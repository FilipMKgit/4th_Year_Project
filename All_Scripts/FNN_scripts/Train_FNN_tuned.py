#this script applies feedforward neural network (fnn) for classification
#it trains the model using noise-free training and validation data, then predicts labels for test data

from pathlib import Path #for handling file operations
import tensorflow as tf #for building the neural network
from sklearn.preprocessing import StandardScaler, LabelEncoder #for data preprocessing
import pandas as pd #for handling dataframes
import numpy as np #for numerical operations
import joblib #for saving encoders and scalers
from collections import Counter #for class balancing

################################################################
#file paths  

#set up paths using pathlib
train_data_path=Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/split_data/Training_data") #training data folder
valid_data_path=Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/split_data/Validation_data") #validation data folder
test_data_path=Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/split_data/Test_data") #test data folder

model_save_path=Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/All_Scripts/fnn_scripts/custom_phyto_model.h5") #fnn model save path
encoder_save_path=Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/All_Scripts/fnn_scripts/FNN_label_encoder.pkl") #label encoder save path
scaler_save_path=Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/All_Scripts/fnn_scripts/FNN_scaler.pkl") #scaler save path

################################################################
#function to load data  

def load_data(directory):
    """reads all csv files from the directory and combines them into a dataframe."""
    csv_files=list(directory.glob("*.csv")) #find all csv files
    dataframes=[pd.read_csv(file) for file in csv_files] #read all csvs
    return pd.concat(dataframes, ignore_index=True) #merge into one dataframe

################################################################
#load training and validation data  

df_train=load_data(train_data_path) #load training data
df_valid=load_data(valid_data_path) #load validation data

################################################################
#feature selection  

selected_features=["FSC-H","SSC-H","FL2-H","FL3-H"] #define features to use

def check_missing_features(df,dataset_name):
    """checks if selected features exist in the dataset."""
    missing=[col for col in selected_features if col not in df.columns] #check for missing features
    if missing: #if features are missing
        print(f"\033[1;31mmissing columns in {dataset_name} dataset: {missing}\033[0m") #print warning
        exit() #terminate script

#check if features exist
for df,name in zip([df_train,df_valid],["training","validation"]): #loop through datasets
    check_missing_features(df,name) #verify features

X_train=df_train[selected_features].values #extract training features
y_train=df_train["Label"].values #extract training labels
X_valid=df_valid[selected_features].values #extract validation features
y_valid=df_valid["Label"].values #extract validation labels

################################################################
#label encoding  

encoder=LabelEncoder() #initialize label encoder
y_train_encoded=encoder.fit_transform(y_train) #encode training labels
y_valid_encoded=encoder.transform(y_valid) #encode validation labels
joblib.dump(encoder,encoder_save_path) #save encoder

################################################################
#feature scaling  

scaler=StandardScaler() #initialize scaler
X_train_scaled=scaler.fit_transform(X_train) #scale training features
X_valid_scaled=scaler.transform(X_valid) #scale validation features
joblib.dump(scaler,scaler_save_path) #save scaler

################################################################
#class weighting  

class_counts=Counter(y_train_encoded) #count occurrences of each class
total_samples=len(y_train_encoded) #total number of samples
class_weights={cls:total_samples/(len(class_counts)*count) for cls,count in class_counts.items()} #calculate weights

################################################################
#build fnn model  

model=tf.keras.Sequential([
    tf.keras.layers.Dense(128,activation="relu",kernel_regularizer=tf.keras.regularizers.l2(0.001),input_shape=(X_train_scaled.shape[1],)), #first dense layer
    tf.keras.layers.BatchNormalization(), #normalize activations
    tf.keras.layers.Dropout(0.5), #dropout for regularization
    tf.keras.layers.Dense(64,activation="relu",kernel_regularizer=tf.keras.regularizers.l2(0.001)), #second dense layer
    tf.keras.layers.BatchNormalization(), #normalize activations
    tf.keras.layers.Dropout(0.5), #dropout for regularization
    tf.keras.layers.Dense(len(encoder.classes_),activation="softmax") #output layer with softmax
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),loss="sparse_categorical_crossentropy",metrics=["accuracy"]) #compile model

################################################################
#train model  

history=model.fit(X_train_scaled,y_train_encoded,
    validation_data=(X_valid_scaled,y_valid_encoded),
    epochs=50,batch_size=32,verbose=1,
    class_weight=class_weights) #train model

model.save(model_save_path) #save trained model

################################################################
#print results  

print("\033[1;32m=======================\n      FNN training completed       \n=======================\033[0m") #print success message
