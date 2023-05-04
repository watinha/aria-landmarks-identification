import pandas as pd, numpy as np, sys, pickle

from sklearn import metrics
from sklearn.model_selection import GroupShuffleSplit

from pipeline.config import get_pipeline

df = pd.read_csv('./data/similarity-dataset.csv')
numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
filtered_columns = [c for c in numeric_columns
                      if c.find('_Unnamed') == -1 and c != 'similar']
base_columns = [c for c in filtered_columns if c.startswith('base_')]
target_columns = [c for c in filtered_columns if c.startswith('target_')]
groups = df['base_url'].to_numpy()

X = np.abs(df[base_columns].to_numpy() - df[target_columns].to_numpy())
y = df['similar'].to_numpy()

classifiers = ['dt', 'rf', 'lsvm', 'knn']
results = {}
for classifier in classifiers:
  results[classifier] = { 'y_true': [], 'y_pred': [] }

_, ncol = X.shape

folds = GroupShuffleSplit(n_splits=10, random_state=42)
for train, test in folds.split(X, y, groups):
  X_train, y_train = X[train, :], y[train]
  X_test, y_test = X[test, :], y[test]
  group_train = groups[train]

  for classifier in classifiers:
    pipeline = get_pipeline(classifier, ncol)
    pipeline.fit(X_train, y_train, groups=group_train)

    results[classifier]['y_pred'].extend(pipeline.predict(X_test))
    results[classifier]['y_true'].extend(y_test)


for classifier in classifiers:
  y_pred = results[classifier]['y_pred']
  y_true = results[classifier]['y_true']
  print('==========================')
  print('-                        -')
  print('-   %s    -' % (classifier))
  print()
  print(metrics.classification_report(y_true, y_pred))
  print()

  pipeline = get_pipeline(classifier, ncol)
  pipeline.fit(X, y, groups=groups)

  with open('./results/classifier/pipeline-similarity-%s.sav' % (classifier), 'wb') as f:
    pickle.dump(pipeline, f)

