import pickle, editdistance, os, pandas as pd, numpy as np

from sklearn import metrics
from sklearn.model_selection import GroupShuffleSplit

from pipeline.config import get_pipeline

XPATH_DISTANCE = 1
THRESHOLD = 0.5
LANDMARKS = ['banner', 'complementary', 'contentinfo', 'form', 'main', 'navigation', 'region', 'search']

def search_regions ():
  pipeline = pickle.load(open('./results/classifier/pipeline-similarity-rf.sav', 'rb'))

  cluster_reports = [ report for report in os.listdir('./results/clusters/') if report.endswith('.csv') and report.startswith('region-') and not report.endswith('.xpath.csv') and not report.endswith('.similar.csv')]
  test_reports = [ report for report in os.listdir('./results/test/') if report.endswith('-test.csv') ]

  test_map = {}
  for filename in test_reports:
    df = pd.read_csv('./results/test/%s' % (filename))

    url_list = df['url'].unique().tolist()
    assert(len(url_list) == 1)
    url = url_list.pop()

    test_map[url] = filename
    del df

  for filename in cluster_reports:
    df = pd.read_csv('./results/clusters/%s' % (filename))

    url_list = df['url'].unique().tolist()
    assert(len(url_list) == 1)
    url = url_list.pop()

    test_df = pd.read_csv('./results/test/%s' % (test_map[url]))

    cluster_xpaths = df['xpath'].tolist()
    test_xpaths = test_df['xpath'].tolist()

    (base_ind, target_ind) = get_similar_xpaths(cluster_xpaths, test_xpaths)
    similar_df = test_df.iloc[target_ind]

    xpath_df = pd.concat([df, similar_df])
    new_name = filename.replace('.csv', '.xpath.csv')
    xpath_df.to_csv('./results/clusters/%s' % (new_name))

    print('%s - xpath - %d -> %d' % (url, df.shape[0], xpath_df.shape[0]))

    classified_df = pd.DataFrame()
    for i, ind in enumerate(target_ind):
      base = df.iloc[[base_ind[i]]]
      target = test_df.iloc[[ind]]
      target.index = base.index
      base.columns = [('base_%s' % (c)) for c in base.columns]
      target.columns = [('target_%s' % (c)) for c in target.columns]
      similarity_df = pd.concat([base, target], axis=1)

      X = features_from_similarity_df(similarity_df)
      y = pipeline.predict(X)

      if y > THRESHOLD:
        classified_df = pd.concat([classified_df, test_df.iloc[[ind]]])

    classified_df = pd.concat([df, classified_df])
    new_name = filename.replace('.csv', '.xpath.similar.csv')
    classified_df.to_csv('./results/clusters/%s' % (new_name))

    print('%s - classified - %d -> %d' % (url, xpath_df.shape[0], classified_df.shape[0]))
    print('')

    del df
    del test_df
    del xpath_df
    del classified_df


def get_similar_xpaths (cluster, test):
  result_base = []
  result_target = []

  for j, xpath1 in enumerate(cluster):
    for i, xpath2 in enumerate(test):
      arr1 = xpath1.split('/')
      arr2 = xpath2.split('/')

      while len(arr1) < len(arr2): arr1.append('')
      while len(arr2) < len(arr1): arr2.append('')

      if editdistance.eval(arr1, arr2) == XPATH_DISTANCE:
        result_base.append(j)
        result_target.append(i)

  return (result_base, result_target)


def similar(baseline, xpaths):
  result = []

  for xpath in xpaths:
    arr1 = baseline.split('/')
    arr2 = xpath.split('/')

    while len(arr1) < len(arr2): arr1.append('')
    while len(arr2) < len(arr1): arr2.append('')

    if editdistance.eval(arr1, arr2) <= XPATH_DISTANCE: result.append(True)
    else: result.append(False)

  return result


def generate_similarity_dataset ():
  df = pd.read_csv('./data/training.classified.csv')
  regions = df.loc[df['class'] == 'region']
  non_regions = df.loc[df['class'] != 'region']

  urls = regions['url'].unique().tolist()

  result = pd.DataFrame()

  for i, url in enumerate(urls):
    print('-------------------------')
    print('URL: %d / %d' % (i, len(urls)))
    print('Number of rows: %d' % (result.shape[0]))

    regions_url = regions.loc[regions['url'] == url]
    non_regions_url = non_regions.loc[non_regions['url'] == url]

    regions_xpath = regions_url['xpath'].tolist()
    for i, xpath in enumerate(regions_xpath):
      similar_xpaths = similar(xpath, regions_xpath)
      similar_regions = regions_url.loc[similar_xpaths]
      start, _ = result.shape
      nrows, _ = similar_regions.shape
      similar_regions.index = [(start + i) for i in range(nrows)]

      base_region = regions_url.iloc[[i]]
      base_regions = base_region.loc[base_region.index.repeat(nrows)]
      base_regions.index = similar_regions.index

      base_regions.columns = [('base_%s' % (c)) for c in base_regions.columns]
      similar_regions.columns = [('target_%s' % (c)) for c in similar_regions.columns]

      similar_rows = pd.concat([base_regions, similar_regions], axis=1)
      similar_rows.loc[:, 'similar'] = 1

      result = pd.concat([result, similar_rows])

    for i, xpath in enumerate(regions_xpath):
      start, _ = result.shape
      nrows, _ = non_regions_url.shape
      base_region = regions_url.iloc[[i]]
      base_regions = base_region.loc[base_region.index.repeat(nrows)]
      base_regions.index = [(start + i) for i in range(nrows)]

      target_regions = non_regions_url.copy()
      target_regions.index = base_regions.index

      base_regions.columns = [('base_%s' % (c)) for c in base_regions.columns]
      target_regions.columns = [('target_%s' % (c)) for c in target_regions.columns]

      different_rows = pd.concat([base_regions, target_regions], axis=1)
      if different_rows.shape[0] > 0: different_rows.loc[:, 'similar'] = 0

      result = pd.concat([result, different_rows])

  result.to_csv('./data/similarity-dataset.csv')


def features_from_similarity_df (df):
  numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
  filtered_columns = [c for c in numeric_columns
                        if c.find('_Unnamed') == -1 and c != 'similar' and 'className' not in c]
  base_landmarks = ['base_%s' % (landmark) for landmark in LANDMARKS]
  base_columns = [c for c in filtered_columns if c.startswith('base_') and c not in base_landmarks]
  target_landmarks = ['target_%s' % (landmark) for landmark in LANDMARKS]
  target_columns = [c for c in filtered_columns if c.startswith('target_') and c not in target_landmarks]

  X = np.abs(df[base_columns].to_numpy() - df[target_columns].to_numpy())
  return X


def fit_similarity_classifier (classifiers):
  df = pd.read_csv('./data/similarity-dataset.csv')
  X = features_from_similarity_df(df)
  y = df['similar'].to_numpy()
  groups = df['base_url'].to_numpy()

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


def cluster_similars (filename, landmark='region'):
    clusters = []
    df_url = pd.read_csv(filename)
    nrows_url, _ = df_url.shape

    for i in range(nrows_url-1, -1, -1):
        target = df_url.iloc[i, :]
        left, right, top, bottom = target['left'], target['left'] + target['width'], target['top'], target['top'] + target['height']
        count = 0

        for cluster in clusters:
            if (top >= cluster['bottom'] or bottom <= cluster['top'] or
                left >= cluster['right'] or right <= cluster['left']): # no insersection
                count = count + 1
            else:
                cluster['left'] = min(left, cluster['left'])
                cluster['right'] = max(right, cluster['right'])
                cluster['top'] = min(top, cluster['top'])
                cluster['bottom'] = max(bottom, cluster['bottom'])

                if cluster['higher_probability'][landmark] < target[landmark]:
                    cluster['higher_probability'] = target
                break

        if count == len(clusters):
            clusters.append({
                'higher_probability': target,
                'left': left,
                'right': right,
                'top': top,
                'bottom': bottom
            })

        for i in range(len(clusters)): # merge clusters
           cluster_1 = clusters[i]
           to_remove = []
           for j in range(i+1, len(clusters)):
               cluster_2 = clusters[j]
               if (cluster_1['top'] >= cluster_2['bottom'] or
                   cluster_1['bottom'] <= cluster_2['top'] or
                   cluster_1['left'] >= cluster_2['right'] or
                   cluster_1['right'] <= cluster_2['left']): # no insersection
                   pass
               else:
                   clusters[i]['left'] = min(clusters[i]['left'], clusters[j]['left'])
                   clusters[i]['right'] = max(clusters[i]['right'], clusters[j]['right'])
                   clusters[i]['top'] = min(clusters[i]['top'], clusters[j]['top'])
                   clusters[i]['bottom'] = max(clusters[i]['bottom'], clusters[j]['bottom'])

                   if (cluster_1['higher_probability'][landmark] < cluster_2['higher_probability'][landmark]):
                       clusters[i]['higher_probability'] = cluster_2['higher_probability']

                   to_remove.append(j)

           to_remove.reverse()
           for remove_index in to_remove:
                clusters.pop(remove_index)

           if i + 1 >= len(clusters): break

    result = pd.DataFrame([])
    for cluster in clusters:
        result = pd.concat([result, pd.DataFrame([cluster['higher_probability']])])

    if (result.size > 0):
      new_filename = filename.replace('.csv', '.clustered.csv')
      result.to_csv(new_filename)


