
# Applications of Machine Learning to the Identification and Characterization of Pico- and Nano-Plankton in Irish Waters
# School of Science and Engineering, University of Galway
# Filip Kłosowski
# 14/03/2025





## Project Description
This project applies machine learning techniques to pico- and nano-plankton flow cytometry data from Irish waters. 
Dimensionality reduction (PCA, t-SNE, UMAP), unsupervised clustering (HDBSCAN, KMeans, GMM), and supervised classification (CNN, LDA, RF and XGB) 
were used to analyze plankton communities. The dataset consists of annotated and unannotated cytometry results. 
This repository contains all scripts necessary to preprocess, analyze, and visualize the data.





## Software
Excel version 2501 (Microsoft), FlowJo version 10_10.0 (BD Biosciences), Python version 3.12.5 (PSF), Visual Studio Code version 1.97.2 (Microsoft)

## Python Libraries
# Data Handling & Manipulation
Pandas, NumPy

# Machine Learning & Clustering
Scikit-learn (StandardScaler, GaussianMixture, KMeans, LinearDiscriminantAnalysis, RandomForestClassifier, XGBoostClassifier), UMAP, t-SNE, HDBSCAN, TensorFlow

# Metrics & Optimization
Silhouette Score, Adjusted Rand Score, Normalized Mutual Information Score, F1 Score, Accuracy Score, Precision Score, Recall Score, Confusion Matrix, Scipy (linear_sum_assignment)

# Visualization
Matplotlib, Seaborn, Matplotlib Colors (to_rgba, Normalize)

# File Handling & Utilities
Pathlib, OS, Joblib, Shutil, Re, Random, Warnings




## Repository Structure
```
📂 4YR_PROJ                                        # Main directory 
│── 📂 All_Scripts                                 # Contains all scripts
│── ├── 📂 Dimensionality_reduction_scripts        # Contains all dimensionality reduction scripts
│── ├── ├──⚙️ make_PCA.py                          # Applies PCA-transformation to files from input dir
│── ├── ├──⚙️ make_tsne.py                         # Applies TSNE-transformation to files from input dir
│── ├── ├──⚙️ make_UMAP.py                         # Applies UMAP-transformation to files from input dir
│── ├── ├──⚙️ make_test_make_tsne_perplexity.py    # Applies TSNE-transformation to test for perplexity= 2, 5, 30, 50, 100
│── ├── ├──⚙️ test_make_UMAP_minDist.py            # Applies UMAP-transformation to test for min_Dist= 2, 5, ,10, 20, 50, 100 200 
│── ├── ├──⚙️ test_make_UMAP_ncomp.py              # Applies UMAP-transformation to test for n_comp= 0.25, 0.5, 0.8, 0.99 
│── ├── ├──📊 viz_test_tnse.py                     # Generates plot for tsne-transformed data
│── ├── ├──📊 viz_test_UMAP.py                     # Generates plot for UMAP-transformed data
│── ├── 📂 FNN_scripts                             # Contains all FNN scripts and models, encoders and scalers
│── ├── ├──🧠 custom_phyto_model.h5                # Saved trained FNN model for classifying plankton population
│── ├── ├──🔄 FNN_label_encoder.pkl                # Converts categorical labels into numerical format
│── ├── ├──🔄 FNN_scaler.pkl                       # Scaler object for normalizing feature values.
│── ├── ├──⚙️ Run_FNN.py                           # Loads and runs the trained FNN model on unannotated test data
│── ├── ├──📈 stat_FNN.py                          # Computes classification metrics: 
│── ├── ├──⚙️ Train_FNN.py                         # Trains the FNN model using labelled cytometry data
│── ├── ├──📊 viz_FNN.py                           # Generates plot for FNN predictions
│── ├── 📂 GMM_scripts                             # Contains all GMM scripts
│── ├── ├──⚙️ make_data_gmm.py                     # Clusters the raw phototroph data using GMM
│── ├── ├──⚙️ make_PCA_gmm.py                      # Clusters the PCA-transformed data using GMM
│── ├── ├──⚙️ make_tSNE_gmm.py                     # Clusters the t-SNE-transformed data using GMM
│── ├── ├──⚙️ make_UMAP_gmm.py                     # Clusters the UMAP-transformed data using GMM
│── ├── ├──📈 stat_gmm.py                          # Computes ARI, NMI, F1-scores and Silhouette scores for all GMM
│── ├── ├──📊 viz_gmm.py                           # Generates plot for GMM clustered data
│── ├── 📂 HDBSCAN_scripts                         # Contains all HDBSCAN scripts
│── ├── ├──⚙️ make_data_hdbscan.py                 # Clusters the raw phototroph data using HDBSCAN
│── ├── ├──⚙️ make_PCA_hdbscan.py                  # Clusters the PCA-transformed data using HDBSCAN
│── ├── ├──⚙️ make_tSNE_hdbscan.py                 # Clusters the t-SNE-transformed data using HDBSCAN
│── ├── ├──⚙️ make_UMAP_hdbscan.py                 # Clusters the UMAP-transformed data using HDBSCAN
│── ├── ├──📈 stat_hdbscan.py                      # Computes ARI, NMI, F1-scores and Silhouette scores for all HDBSCAN data
│── ├── ├──📊 viz_hdbscan.py                       # Generates plot for HDBSCAN clustered data
│── ├── 📂 Kmeans_scripts                          # Contains all k-means scripts
│── ├── ├──⚙️ make_data_kmeans.py                  # Clusters the raw phototroph data using k-means
│── ├── ├──⚙️ make_PCA_kmeans.py                   # Clusters the PCA-transformed data using k-means
│── ├── ├──⚙️ make_tSNE_kmeans.py                  # Clusters the t-SNE-transformed data using k-means
│── ├── ├──⚙️ make_UMAP_kmeans.py                  # Clusters the UMAP-transformed data using k-means
│── ├── ├──📈 stat_kmeans.py                       # Computes ARI, NMI, F1-scores and Silhouette scores for all k-means data
│── ├── ├──📊 viz_elbow_plot_for.py                # Generates an elbow plot to determine the optimal value for k
│── ├── ├──📊 viz_kmeans.py                        # Generates plot for k-means clustered data
│── ├── 📂 LDA_scripts                             # Contains all LDA scripts
│── ├── ├──⚙️ train+run_LDA.py                     # Trains LDA model and runs it on test data to predict clusters
│── ├── ├──📈 stat_LDA.py                          # Computes ARI, NMI and F1-scores for all LDA predictions
│── ├── ├──📊 viz_LDA.py                           # Generates plot for predicted LDA clusters
│── ├── 📂 Pre-processing                          # Contains pre-processing and misc. scripts
│── ├── ├──⚙️ make_annotated_data_with_noise.py    # Compares raw data with annotated data to obain annotated data with noise
│── ├── ├──⚙️ make_annotated_data.py               # Combines gates to obtain labelled data
│── ├── ├──⚙️ make_split_data.py                   # Splits raw and annotated data into training, validation and test data dir
│── ├── ├──⚙️ make_standardized_data.py            # Standardizes all raw data
│── ├── ├──📊 viz_annotated_data_with_noise.py     # Generates plot for annotated data including noise
│── ├── ├──📊 viz_annotated_data.py                # Generates plot for annotated data excluding noise
│── ├── 📂 RF_scripts                              # Contains RF and XGB scripts
│── ├── ├──⚙️ train+run_RF.py                      # Trains RF model and runs it on test data to predict clusters
│── ├── ├──⚙️ train+run_XGB.py                     # Trains XGB model and runs it on test data to predict clusters
│── ├── ├──📈 stat_RF.py                           # Computes accuracy, precision, recall and confusion matrix for RF predictions
│── ├── ├──📈 stat_XGB.py                          # Computes accuracy, precision, recall and confusion matrix for XGB predictions
│── ├── ├──📊 viz_RF_XGB.py                        # Generates plot for predicted RF and XGB clusters
│── 🗑️ Bin                                         # Bin directory
│── 📂 raw_data                                    # Contains raw files, i.e. starting point of project
│── ├── ├──📂 fcsdata                              # Contains all FCS files imported from directly FlowJo
│── ├── ├──📂 fcsdata_photo                        # Contains gated phototrophs CSV files imported directly from FlowJo
│── ├── ├──📂 gates                                # Contains CSV files for phototrophs, Syn, Crypto, Nano1/2 and PEuk1/2
│── ├── ├──📂 phototrophs                          # Cotains CSV phototroph files
│── 🖥️ sklearn-env                                 # Virtual environment containing dependencies and installed packages  
│── 📂 work_data                                   # Contains processed datasets (main output folder)
│── ├── ├──📂 annotated_data                       # Output of make_annotated_data.py
│── ├── ├──📂 annotated_data_with_noise            # Output of make_annotated_data_with_no_noise.py
│── ├── ├──📂 FNN_data                             # Output of Run_FNN.py
│── ├── ├──📂 GMM_data                             # Outputs of make_gmm.py scripts
│── ├── ├──📂 HDBSCAN_data                         # Outputs of make.hdbscan.py scripts
│── ├── ├──📂 k-means_data                         # Outputs of make_kmeans.py scripts
│── ├── ├──📂 LDA_data                             # Output of train_run_LDA.py
│── ├── ├──📂 PCA_data                             # Output of make_PCA.py
│── ├── ├──🗑️ preprocessed_data_no_gates_bin       # Bin for CSV files that had no corresponding gates, could not be annotated
│── ├── ├──📂 RandomForest_data                    # Outputs of train+run_RF.py and train_run_XGB.py
│── ├── ├──📂 split_data                           # Output of make_split_data.py
│── ├── ├── ├──📂 Test_data                        # Unnanotated test files for testing models (15%)
│── ├── ├── ├──📂 Test_data_annotated              # Annotated test files corresponding to unannotated test files for evaluation
│── ├── ├── ├──📂 Training_data                    # Training annotated data for training models (70%)
│── ├── ├── ├──📂 Training_data_no_noise           # Corresponding training annotated data without noise for training models 
│── ├── ├── ├──📂 Validation_data                  # Validation annotated data used for training models (15%)
│── ├── ├── ├──📂 Validation_data_no_noise         # Corresponding annotated validation data for training models with no noise
│── ├── ├──📂 tSNE_data                            # Output of make_tsne.py
│── ├── ├──📂 tSNE_test_data                       # Output of make_test_make_tsne_perplexity.py
│── ├── ├──📂 UMAP_data                            # Output of make_umap.py
│── ├── ├──📂 UMAP_test_data                       # Outputs of test_make_UMAP_minDist.py and test_make_UMAP_ncomp.py
│── ├── ├──📂 working_data_standardized            # Output of make_standardized_data.py
│── ├── ├──📂 working_preprocessed_data            # Manually selected CSV files for processing copied from raw_data/phototrophs
│── 📖 README.md                                   # This README.md file
│── 📖 requirements.txt                            # For library installation    (pip install -r requirements.txt )      
```



## Usage
### Main Pipeline:
This is the order of files ran to get the study's results, and thus it is recommended that the user follows this order to replicate the results (note the split_data script randomizes the file allocation, thus results of supervised models will vary):

First run 'pip install -r requirements.txt' to install all dependencies from requirements.txt

1. ⚙️ make_annotated_data.py
2. ⚙️ make_annotated_data_with_noise.py
3. ⚙️ make_standardized_data.py
4. ⚙️ make_split_data.py                          
5. ⚙️ make_PCA.py
6. ⚙️ make_test_make_tsne_perplexity.py
7. ⚙️ make_tsne.py
8. ⚙️ test_make_UMAP_minDist.py
9. ⚙️ test_make_UMAP_ncomp.py
10. ⚙️ make_UMAP.py
11. 📊 viz_elbow_plot_for.py
12. ⚙️ make_data_kmeans.py
13. ⚙️ make_PCA_kmeans.py
14. ⚙️ make_tSNE_kmeans.py
15. ⚙️ make_UMAP_kmeans.py
16. ⚙️ make_data_hdbscan.py
17. ⚙️ make_PCA_hdbscan.py
18. ⚙️ make_tSNE_hdbscan.py
19. ⚙️ make_UMAP_hdbscan.py
20. ⚙️ make_data_gmm.py
21. ⚙️ make_PCA_gmm.py
22. ⚙️ make_tSNE_gmm.py
23. ⚙️ make_UMAP_gmm.py
24. ⚙️ train+run_LDA.py
25. ⚙️ Train_FNN.py
26. ⚙️ Run_FNN.py
27. ⚙️ train+run_RF.py
28. ⚙️ train+run_XGB.py

#All the neccessary output files can be found in associated directories under the work_data directory
#To visualize the output files on plots corresponding 📊viz.py should be run
#To evaluate the output results the corresponding 📈stat.py should be run





## Credits
This project was conducted as part of my undergraduate thesis in Marine Science at the University of Galway.

It was developed using scikit-learn for machine learning implementations and refined with debugging assistance from Stack Overflow and ChatGPT.
