from pathlib import Path #for handling file operations
import pandas as pd #for data manipulation
import numpy as np #for numerical operations
import random #for selecting random files
import matplotlib.pyplot as plt #for plotting
from matplotlib.colors import to_rgba #for color mapping

################################################################
#set directories
main_dir = Path("C:/Users/filip/OneDrive/Desktop/4YR_PROJ/work_data/UMAP_test_data")

################################################################
#choice 1 - select UMAP test type (n_neighbors or min_dist)
def pickUMAPTestType():
    print("\nChoose UMAP test type:")
    print("1. n_neighbors")
    print("2. min_dist")
    choice = input("Enter 1 or 2: ").strip()
    
    return "n_neighbors" if choice == "1" else "min_dist"

################################################################
#choice 2 - choose UMAP parameter folder
def pickUMAPFolder(umap_test_type):
    prefix = "UMAP_test_n=" if umap_test_type == "n_neighbors" else "UMAP_test_minDist="
    umap_folders = sorted([f for f in main_dir.iterdir() if f.is_dir() and f.name.startswith(prefix)], key=lambda x: float(x.name.split("=")[-1]))
    
    print(f"\nAvailable {umap_test_type} values:")
    for i, folder in enumerate(umap_folders, start=1):
        print(f"{i}. {folder.name}")
    
    choice = input(f"Pick a {umap_test_type} folder: ").strip()
    return umap_folders[int(choice) - 1] if choice.isdigit() and 1 <= int(choice) <= len(umap_folders) else umap_folders[0]

################################################################
#choice 3 - choose file
def pickFile(folder_path):
    csvFiles = list(folder_path.glob("*.csv"))
    
    print("\n1. Choose a specific file")
    print("2. Pick a random file")
    user_choice = input("Enter 1 or 2: ").strip()
    
    if user_choice == "2":
        return random.choice(csvFiles)
    
    for i, file in enumerate(csvFiles, start=1):
        print(f"{i}. {file.name}")
    
    selection = input("Enter file number: ").strip()
    return csvFiles[int(selection) - 1] if selection.isdigit() and 1 <= int(selection) <= len(csvFiles) else csvFiles[0]

################################################################
#plot UMAP clusters with mapped Labels
def plot_umap(filePath):
    df = pd.read_csv(filePath)
    
    if not {'UMAP1', 'UMAP2', 'Cluster', 'Label'}.issubset(df.columns):
        print("Error: The selected file does not contain the necessary UMAP transformed data.")
        return
    
    cluster_mapping = dict(zip(df["Cluster"], df["Label"]))
    noise = df[df["Cluster"] == -1]
    species = df[df["Cluster"] != -1]
    
    unique_clusters = sorted(species["Cluster"].unique())
    cmap = plt.get_cmap("turbo")
    cluster_colors = {cluster: cmap(i / max(1, len(unique_clusters) - 1)) for i, cluster in enumerate(unique_clusters)}

    plt.figure(figsize=(8, 6))
    
    if not noise.empty:
        plt.scatter(noise["UMAP1"], noise["UMAP2"], c="gray", alpha=0.3, label="Noise", s=10)
    
    for cluster in unique_clusters:
        cluster_points = species[species["Cluster"] == cluster]
        plt.scatter(
            cluster_points["UMAP1"], cluster_points["UMAP2"],
            color=to_rgba(cluster_colors[cluster]), label=cluster_mapping.get(cluster, f"Cluster {cluster}"),
            alpha=0.8, s=15
        )
    
    plt.title(f"UMAP Visualization: {filePath.name}")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Species")
    plt.tight_layout()
    plt.show()

################################################################
#execution of script
if __name__ == "__main__":
    umap_test_type = pickUMAPTestType()
    umap_folder = pickUMAPFolder(umap_test_type)
    file_name = pickFile(umap_folder)
    file_path = umap_folder / file_name
    
    df_data = pd.read_csv(file_path)
    plot_umap(file_path)
