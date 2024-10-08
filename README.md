# Stage 2: Data Preparation and Clustering Analysis

## Overview
This repository contains the deliverables for Stage 2 of the project, focusing on preparing the data for analysis and modeling, ensuring data integrity, and performing clustering analysis to identify distinct customer segments. The main objectives are to preprocess the dataset, checking for duplicates, checking for missing values split it into training and testing sets, apply scaling techniques, and conduct clustering analysis to extract meaningful insights.

## Deliverables

### 2-1: Data Preparation
The data preparation phase is crucial for ensuring that the dataset is clean, structured, and ready for modeling. The following tasks have been completed:

- **Preprocessed Dataset:** 
  - addressed the duplication of the data into multiple rows and removed
  - Addressed missing data points.
  - Encoded categorical variables.
  - The cleaned and preprocessed dataset is available in the `Data_Preparation` folder.

- **Training and Testing Sets:** 
  - The dataset has been split into training and testing sets.
  - These sets are essential for validating the model's performance.
  - Files for training and testing sets are included in the `Data_Preparation` folder with accompanying documentation.

- **Scaling Techniques:** 
  - Applied appropriate scaling techniques to normalize the data.
  - Documentation of the scaling techniques, including relevant code snippets, is provided in the `Data_Preparation` folder.

### 2-2: Clustering Analysis
Clustering analysis was performed to identify distinct customer segments within the dataset. The following steps were carried out:

- **Optimal Number of Clusters:**
  - The optimal number of clusters was determined using the elbow method.
  - The analysis and visualizations supporting the determination of the optimal number of clusters are included in the `Clustering_Analysis` folder.

- **K-Means Clustering Model:**
  - A K-Means clustering model was trained on the dataset.
  - The trained model, along with the code used for training, is available in the `Clustering_Analysis` folder.

- **Visualization and Labeling of Clusters:**
  - Visualizations of the resulting clusters are provided to facilitate interpretation.
  - Each cluster has been labeled, and insights derived from the analysis are documented in the `Clustering_Analysis` folder.

## Repository Structure
The repository is organized into two main folders to maintain clarity and organization:

- **Data_Preparation:** Contains all deliverables related to data preprocessing, including the preprocessed dataset, training and testing sets, and scaling techniques documentation.

- **Clustering_Analysis:** Contains all deliverables related to clustering analysis, including the optimal number of clusters, trained K-Means model, and visualizations with labels.


This README file outlines the deliverables for Stage 2 and provides clear instructions on how the project is structured and organized. For any questions or further information, please feel free to contact the project team.
