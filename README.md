=============================
ARIA-Landmarks Identification
=============================

This project store scripts for the evaluation of an approach for automatically identifying ARIA-Landmarks in web applications. The approach is based on DOM elements classification and clustering for identifying elements with the highest probability of being ARIA Landmarks.

The project was implemented in Python and presents a Dockerfile for generating an Docker image capable of running the scripts.

The main file of the project (main.py) executes multiple activities for conducting the identification of ARIA-Landmarks. The activities and their input/output resources are described next:
1. **Classifier training (pipeline.cross_validation.fit_classifier)**: trains a classifier (SVM, KNN, DT or RF) using the training dataset (./data/training.classified.csv) and saves the generated extractor and classifier pickled files in ;/results/classifier folder. The scripts also runs 10-fold CV for generating accuracy reports for the classifier.
2. **Merge CV reports (pipeline.merge_cv_reports.merge_reports)**: merge the CV accuracy reports generated in the previous activity into a single spreadsheet for analysis (./results/accuracy.xlsx). This file contains accuracy reports considering F1-Score/Precision and Recall for each class, average macro and weighted average values. The spreadsheet also presents the frequency that each feature of the classification models were used to compose the extractor/classifier pair.
3. **Classify test dataset (pipeline.classify_test_dataset)**: uses the fitted RF extractor/classifier (./results/classifier) for classifying the samples available in the test dataset (./data/test/). The test dataset is composed of data extracted from elements of different web applications. The results of this activity are stored in the ./resutls/test folder.
4. **Clustering test dataset predictions (pipeline.clustering_rows.cluster_rows)**: cluster the results of the test dataset classification accordingly to their class denomination and position/size features. For each cluster, only the element with the highest probability of being an ARIA landmark is reported. The results of this activity are stored in the ./results/clusters folder.
5. **Generating image reports (pipeline.image_report.generate_reports)**: generates image reports for the ARIA landmarks identified in the previous activity (./results/clusters folder) and the screenshots of the respective web applications (./data/screenshots folder). Image reports are generated for each ARIA landmark identified in the previous activities and are stored in the ./results/image-reports folder.

