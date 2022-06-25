from pipeline.cross_validation import fit_classifier
from pipeline.merge_cv_reports import merge_reports
from pipeline.classify_test_dataset import classify_test
from pipeline.clustering_rows import cluster_rows
from pipeline.image_report import generate_reports

classifiers = ['svm', 'knn', 'dt', 'rf']
for classifier in classifiers:
    print('fitting %s' % (classifier))
    fit_classifier(classifier)
    print('results saved in ./results/classifier folder')

print('')
merge_reports()
print('cv classifier reports merged in ./results/cv')

print('')
print('predicting test dataset...')
classify_test()
print('test classification results saved in ./results/cv')

print('')
print('clustering rows according to classification and position/size...')
cluster_rows()
print('clustering results saved in ./results/clusters')

print('')
print('generating image reports...')
generate_reports()
print('image reports saved in ./results/image-reports')

