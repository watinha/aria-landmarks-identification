import pandas as pd, numpy as np, sys

from sklearn.model_selection import GroupShuffleSplit

from pipeline.config import get_pipeline, EnhancingFeatures

df = pd.read_csv('./data/similarity-dataset.csv')
numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
filtered_columns = [c for c in numeric_columns
                      if c.find('_Unnamed') == -1 and c != 'similar']
base_columns = [c for c in filtered_columns if c.startswith('base_')]
target_columns = [c for c in filtered_columns if c.startswith('target_')]
groups = df['base_url'].to_numpy()

X = np.abs(df[base_columns].to_numpy() - df[target_columns].to_numpy())
y = df['similar'].to_numpy()

y_pred = []
y_true = []

classifier_name = 'rf'
_, ncol = X.shape

folds = GroupShuffleSplit(n_splits=10, random_state=42)
for train, test in folds.split(X, y, groups):
  X_train, y_train = X[train, :], y[train]
  X_test, y_test = X[test, :], y[test]
  group_train = groups[train]

  pipeline = get_pipeline(classifier_name, ncol)
  pipeline.fit(X_train, y_train, groups=group_train)

  y_pred.extend(pipeline.predict(X_test))
  y_true.extend(y_test)

print(metrics.classification_report(y_true, y_pred))
