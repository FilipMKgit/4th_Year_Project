#this script applies xgboost for classification
#it trains the model using noise-free training and validation data, then predicts labels for test data

from pathlib import Path #for handling file operations
import pandas as pd #for dataframe operations
import numpy as np #for numerical operations
import warnings #for suppressing xgboost warnings
import matplotlib.pyplot as plt #for potential visualization
from xgboost import XGBClassifier #for xgboost model
from sklearn.preprocessing import LabelEncoder #for encoding categorical labels

################################################################
#defines input and output directories  

#training and validation data directories (noise-free)
train_folder = Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/split_data/Training_data_no_noise") #training data folder
validation_folder = Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/split_data/Validation_data_no_noise") #validation data folder

#test data and output directory
test_folder = Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/split_data/Test_data") #test data folder
output_folder = Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/XGBoost_data/XGBoost_Predictions") #folder for xgboost output

#ensure output folder exists
output_folder.mkdir(parents=True, exist_ok=True) #create folder if it doesn't exist

#feature set for xgboost
FEATURE_COLUMNS = ["FSC-H", "SSC-H", "FL1-H", "FL2-H", "FL3-H"] #selected feature columns

#class label mapping
LABEL_MAP = {"Crypto": 1, "Nano1": 2, "Nano2": 3, "PEuk1": 4, "PEuk2": 5, "Syn": 6, "noise": -1} #map labels to numeric values

#reverse mapping for readable labels
REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()} #reverse dictionary for easy lookup

################################################################
#function to load data from a folder  

def load_data(folder):
    #reads all csv files from the directory and combines them into a dataframe
    all_files = [f for f in folder.glob("*.csv")] #find all csv files
    dataframes = [pd.read_csv(file) for file in all_files] #read all csvs
    return pd.concat(dataframes, ignore_index=True) #combine into one dataframe

################################################################
#train xgboost model  

def train_xgb_model(X, y):
    #trains an xgboost model and returns the fitted model
    xgb_model = XGBClassifier(
        eval_metric='mlogloss', 
        n_estimators=100, 
        max_depth=4, 
        learning_rate=0.1, 
        random_state=42
    ) #initialize xgboost model
    xgb_model.fit(X, y) #train model
    return xgb_model #return trained model

################################################################
#assign cluster numbers  

def assign_clusters(df):
    #assigns numeric values to 'Predicted_Label' to create 'Cluster' column
    if "Predicted_Label" in df.columns: #check if label column exists
        df["Cluster"] = df["Predicted_Label"].map(LABEL_MAP).fillna(-1).astype(int) #assign cluster numbers

################################################################
#process and predict test data  

def process_test_file(filename, model):
    #loads test file, applies xgboost, assigns clusters, and saves output
    filepath = test_folder / filename #define file path
    test_data = pd.read_csv(filepath) #read test file
    
    #apply xgboost to test data
    X_test = test_data[FEATURE_COLUMNS] #extract feature columns
    predicted_labels = model.predict(X_test) #make predictions
    
    #convert predictions back to original labels
    predicted_labels_decoded = [REVERSE_LABEL_MAP[label] for label in predicted_labels] #decode class labels
    
    #add predictions to test data
    test_data["Predicted_Label"] = predicted_labels_decoded #store predicted class

    #assign clusters
    assign_clusters(test_data) #assign cluster numbers

    #define output file path
    output_filename = f"{filename.stem}_XGBoost_predicted.csv" #modify filename
    output_file_path = output_folder / output_filename #define output path

    #save results
    test_data.to_csv(output_file_path, index=False) #save as csv
    
    print(f"Processed {filename.name} -> Saved as {output_filename}") #print success message

################################################################
#load training and validation data  

print("Loading training and validation data...") #print status

train_data = load_data(train_folder) #load training data
validation_data = load_data(validation_folder) #load validation data

#combine both datasets for xgboost training
combined_data = pd.concat([train_data, validation_data], ignore_index=True) #merge datasets
print(f"Combined dataset loaded: {len(combined_data)} rows") #print dataset size

#extract features and labels
y_train = combined_data["Label"].map(LABEL_MAP) #get target labels
X_train = combined_data[FEATURE_COLUMNS] #get feature matrix

#train xgboost model
warnings.filterwarnings("ignore", category=UserWarning, message=".*Parameters: { \"use_label_encoder\" }.*") #suppress xgboost warnings
print("Training XGBoost model...") #print status
xgb_model = train_xgb_model(X_train, y_train) #train model
print("Model training complete.") #print completion message

#process test files
print("\nScanning test folder...") #print status
test_files = [f for f in test_folder.glob("*.csv")] #get test files

print(f"Found {len(test_files)} test files. Processing...") #print number of files
for test_file in test_files: #loop through test files
    process_test_file(test_file, xgb_model) #process each file
print("All test files processed.") #print completion message
