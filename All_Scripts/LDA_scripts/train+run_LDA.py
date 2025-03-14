#this script applies Linear Discriminant Analysis (LDA) for classification
#it trains the model using noise-free training and validation data, then predicts labels for test data

from pathlib import Path #import modern file handling
import pandas as pd #import dataframe operations
import numpy as np #import numerical operations
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis #import LDA model

################################################################
#defines input and output directories  

train_folder=Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/split_data/Training_data_no_noise") #training data folder
validation_folder=Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/split_data/Validation_data_no_noise") #validation data folder
test_folder=Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/split_data/Test_data") #test data folder
output_folder=Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/LDA_data") #folder for LDA output
output_folder.mkdir(parents=True,exist_ok=True) #create folder if it doesn't exist


FEATURE_COLUMNS=["FSC-H","SSC-H","FL1-H","FL2-H","FL3-H"] #selected feature columns

################################################################
#class label mapping
LABEL_MAP={"Crypto":1,"Nano1":2,"Nano2":3,"PEuk1":4,"PEuk2":5,"Syn":6} #map labels to numeric values
REVERSE_LABEL_MAP={v:k for k,v in LABEL_MAP.items()} #reverse dictionary for easy lookup
CONFIDENCE_THRESHOLD=0.2 #minimum confidence score to keep a prediction

################################################################
#function to load data from a folder  
def load_data(folder):
    all_files=[f for f in folder.glob("*.csv")] #find all csv files
    combined_data=[pd.read_csv(file) for file in all_files] #read and store all files
    return pd.concat(combined_data,ignore_index=True) #combine into one dataframe

################################################################
#train
def train_lda_model(X,y):
    lda=LinearDiscriminantAnalysis() #initialize LDA model
    lda.fit(X,y) #train model
    return lda #return trained model

################################################################
#process and predict test data with confidence scores  
def process_test_file(filename,model):
    filepath=test_folder/filename #define file path
    test_data=pd.read_csv(filepath) #read test file
    
    
    #ensure all required columns are present
    missing_cols=set(FEATURE_COLUMNS)-set(test_data.columns) #check for missing columns
    if missing_cols: #if any columns are missing
        print(f"Skipping {filename}: Missing columns {missing_cols}") #print warning
        return #skip file
    
    #apply LDA to test data
    X_test=test_data[FEATURE_COLUMNS] #extract feature columns
    predicted_labels=model.predict(X_test) #make predictions
    predicted_probs=model.predict_proba(X_test) #get confidence scores

    #get highest confidence score for each prediction
    max_confidence=np.max(predicted_probs,axis=1) #extract max confidence per row

    #assign 'Noise' if confidence is low
    final_assignment=[ label if confidence>=CONFIDENCE_THRESHOLD else "Noise" #iterate through data and apply noise threshold
        for label,confidence in zip(predicted_labels,max_confidence)]

    #add predictions and confidence scores to test data
    test_data["Predicted_Label"]=predicted_labels #store predicted class
    test_data["Max_Confidence"]=max_confidence #store confidence score
    test_data["Label"]=final_assignment #store final label assignment

    #define output file path
    output_filename=f"{filename.stem}_LDA_predicted_with_confidence.csv" #modify filename
    output_file_path=output_folder/output_filename #define output path

    #save results
    test_data.to_csv(output_file_path,index=False) #save as csv
    
    print(f"Processed {filename.name} -> Saved as {output_filename}") #print success message

################################################################
#load training and validation data  
print("Loading training and validation data...") #print status
train_data=load_data(train_folder) #load training data
validation_data=load_data(validation_folder) #load validation data

#combine both datasets for LDA training
combined_data=pd.concat([train_data,validation_data],ignore_index=True) #merge datasets
print(f"Combined dataset loaded: {len(combined_data)} rows") #print dataset size

#extract features and labels
y_train=combined_data["Label"] #get target labels
X_train=combined_data[FEATURE_COLUMNS] #get feature matrix

#train LDA model
print("Training LDA model...") #print status
lda_model=train_lda_model(X_train,y_train) #train model
print("Model training complete.") #print completion message

#process test files
print("\nScanning test folder...") #print status
test_files=[f for f in test_folder.glob("*.csv")] #get test files

print(f"Found {len(test_files)} test files. Processing...") #print number of files
for test_file in test_files: #loop through test files
    process_test_file(test_file,lda_model) #process each file
print("All test files processed.") #print completion message
