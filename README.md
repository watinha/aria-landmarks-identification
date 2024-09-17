ARIA-Landmarks Identification
=============================

This project store scripts for the evaluation of an approach for automatically identifying ARIA-Landmarks in web applications. The approach is based on DOM elements classification and clustering for identifying elements with the highest probability of being ARIA Landmarks.

The project was implemented in Python and presents a Dockerfile for generating an Docker image capable of running the scripts.

The main file of the project (main.py) executes multiple activities for conducting the identification of ARIA-Landmarks. The activities and their input/output resources are described next:
1. **Classifier training (pipeline.cross_validation.fit_classifier)**: trains a classifier (SVM, KNN, DT or RF) using the training dataset (./data/training.classified.csv) and saves the generated extractor and classifier pickled files in ;/results/classifier folder. The scripts also runs 10-fold CV for generating accuracy reports for the classifier.
2. **Generating the similarity dataset (pipeline.search_regions)**: generates a dataset with data extracted from ARIA Regions and non-Regions elements from the training dataset. Each element is compared to all other elements in the same website. If both elements are ARIA Regions associated with one another according to their X-Path distance, the similarity is set to 1. Otherwise, the similarity is set to 0.
3. **Training the similarity classifier (pipeline.search_regions)**: trains a classifier (SVM, KNN, DT or RF) using the similarity dataset and saves the generated extractor and classifier. The scripts also runs 10-fold CV for generating accuracy reports for the classifier.
4. **Predicting ARIA Regions (pipeline.search_regions)**: predicts ARIA Regions in the test dataset using the ARIA Landmarks classifier, X-Path distance and the similarity classifier.
5. **Clustering the ARIA Regions (pipeline.search_regions)**: clusters the predicted ARIA Regions considering the DOM hierarchy among regions and the similarity between them.
6. **Reporting the results (pipeline.search_regions)**: generates a report with the identified ARIA Regions and their clusters.
