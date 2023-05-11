from pipeline.regions_search import fit_similarity_classifier

classifiers = ['dt', 'rf', 'lsvm', 'knn']
fit_similarity_classifier(classifiers)
