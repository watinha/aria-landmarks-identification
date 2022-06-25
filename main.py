from pipeline.cross_validation import fit_classifier
from pipeline.merge_cv_reports import merge_reports
from pipeline.classify_test_dataset import classify_test
from pipeline.clustering_rows import cluster_rows

classifiers = ['svm', 'knn', 'dt', 'rf']
for classifier in classifiers:
    print('fitting %s' % (classifier))
    fit_classifier(classifier)
    print('results saved in ./results/classifier folder')

print('')
merge_reports()
print('cv classifier reports merged in ./results/cv')

print('')
classify_test()
print('test classification results saved in ./results/cv')

print('')
cluster_rows()
print('clustering results saved in ./results/clusters')

