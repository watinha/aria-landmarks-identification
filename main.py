from pipeline.classification_pipeline import fit_classifier
from pipeline.merge_classifier_reports import merge_reports

classifiers = ['svm', 'knn', 'dt', 'rf']
for classifier in classifiers:
    print('fitting %s' % (classifier))
    fit_classifier(classifier)
    print('results saved on ./results/classifier folder')

print('')
merge_reports()
print('cv classifier reports merged in ./results/cv')


