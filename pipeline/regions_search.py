import editdistance, os, pandas as pd

XPATH_DISTANCE = 1

def search_regions ():
  cluster_reports = [ report for report in os.listdir('./results/clusters/') if report.endswith('.csv') and report.startswith('region-') and not report.endswith('.xpath.csv') ]
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

    similar_xpaths = get_similar_xpaths(cluster_xpaths, test_xpaths)
    similar_df = test_df.iloc[similar_xpaths]

    new = pd.concat([df, similar_df])
    new_name = filename.replace('.csv', '.xpath.csv')
    new.to_csv('./results/clusters/%s' % (new_name))

    print('%s - %d -> %d' % (url, df.shape[0], new.shape[0]))

    del df
    del test_df
    del new


def get_similar_xpaths (cluster, test):
  result = []

  for xpath1 in cluster:
    for i, xpath2 in enumerate(test):
      arr1 = xpath1.split('/')
      arr2 = xpath2.split('/')

      while len(arr1) < len(arr2): arr1.append('')
      while len(arr2) < len(arr1): arr2.append('')

      if editdistance.eval(arr1, arr2) == XPATH_DISTANCE:
        result.append(i)

  return result


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
