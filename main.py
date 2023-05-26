import os

from pipeline.classify_test_dataset import classify_test
from pipeline.clustering_rows import cluster_rows
from pipeline.cross_validation import fit_classifier
from pipeline.image_report import generate_reports
from pipeline.merge_cv_reports import merge_reports
from pipeline.regions_search import search_regions, fit_similarity_classifier, cluster_similars

classifiers = ['svm', 'knn', 'dt', 'rf']
for classifier in classifiers:
    print('fitting %s' % (classifier))
    fit_classifier(classifier)
    print('results saved in ./results/classifier folder')

print('')

for classifier in classifiers:
    print('fitting similarity %s' % (classifier))
    fit_similarity_classifier(classifier)
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
print('searching for regions using xpath and classifying similar rows...')
search_regions()
print('new regions saved in ./results/clusters')

print('')
print('clustering similar regions...')
similar_filenames = [ './results/clusters/%s' % (f)
    for f in os.listdir('./results/clusters')
    if f.endswith('.xpath.similar.csv') ]
for filename in similar_filenames:
  cluster_similars(filename, landmark='region')
print('clustering results saved in ./results/clusters')

print('')
print('generating image reports...')
generate_reports()
print('image reports saved in ./results/image-reports')
