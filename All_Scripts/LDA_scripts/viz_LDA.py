from pathlib import Path#import modern file handling
import pandas as pd#import dataframe operations
import matplotlib.pyplot as plt#import for plotting
import numpy as np#import numerical operations
import random#import for random selection

################################################################
#set directories  

lda_output_dir=Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/LDA_data")

################################################################
#choice 1 - select file  

def get_lda_files():
    #Retrieve list of available LDA output files
    return [f.name for f in lda_output_dir.glob("*_LDA_predicted_with_confidence.csv")]

def pickFile():
    #Allows user to select an LDA file or pick one randomly
    files=get_lda_files()
    if not files:
        print("No LDA files found.")
        return None
    
    print("\n1. Choose a specific file")
    print("2. Pick a random file")
    user_choice=input("Enter 1 or 2: ").strip()
    if user_choice=="2":
        return random.choice(files)
    if user_choice=="1":
        for i,file in enumerate(files,1):
            print(f"{i}. {file}")
        while True:
            selection=input("Enter file number: ").strip()
            if selection.isdigit():
                idx=int(selection)-1
                if 0<=idx<len(files):
                    return files[idx]
    return files[0]

################################################################
#choice 2 - choose plot parameters  

def pick_plot_columns(df):
    #Allows user to choose plot axes
    print("\nChoose plot parameters:")
    print("1. FSC-H vs SSC-H")
    print("2. FL3-H vs FL2-H")
    print("3. Custom selection")
    user_input=input("Enter number: ").strip()
    if user_input=="1":
        return "FSC-H","SSC-H"
    elif user_input=="2":
        return "FL3-H","FL2-H"
    elif user_input=="3":
        cols=[col for col in df.columns if col not in ["Time","Label","Max_Confidence","Predicted_Label"]]
        for i,col in enumerate(cols,1):
            print(f"{i}. {col}")
        try:
            x_idx=int(input("X-axis column number: "))-1
            y_idx=int(input("Y-axis column number: "))-1
            return cols[x_idx],cols[y_idx]
        except (ValueError,IndexError):
            return "FSC-H","SSC-H"
    return "FSC-H","SSC-H"

################################################################
#graph the LDA output  

################################################################
#graph the LDA output  

def plot_lda_clusters(filePath, x, y):
    #Plots LDA classification results
    df = pd.read_csv(filePath)  # Load data
    df["Label"] = df["Label"].astype(str)  # Ensure labels are strings

    # Apply log transformation to avoid log10(0) issues
    df[x] = np.log10(df[x].replace(0, np.nan))
    df[y] = np.log10(df[y].replace(0, np.nan))
    df.dropna(subset=[x, y], inplace=True)  # Remove NaNs

    # Generate unique colors for each label using Turbo colormap
    unique_labels = sorted(df['Label'].unique())
    turbo_cmap = plt.get_cmap("turbo", len(unique_labels))
    label_colors = {label: turbo_cmap(i / len(unique_labels)) for i, label in enumerate(unique_labels)}

    # Plot clusters
    plt.figure(figsize=(8, 6))
    for label in unique_labels:
        subset_df = df[df['Label'] == label]
        plt.scatter(
            subset_df[x], subset_df[y],
            color=label_colors[label], label=label, alpha=0.7, s=15
        )

    # Customize the plot
    plt.xlabel(f"log10({x})")
    plt.ylabel(f"log10({y})")
    plt.title(Path(filePath).name)
    plt.legend(title="Labels", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

########################################################################
#execution of script  

if __name__=="__main__":
    print("LDA-Based Cluster Visualization")
    file_name=pickFile()
    if file_name:
        file_path=lda_output_dir/file_name
        df_data=pd.read_csv(file_path)
        x_col,y_col=pick_plot_columns(df_data)
        plot_lda_clusters(file_path,x_col,y_col)
