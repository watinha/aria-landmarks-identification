import pandas as pd, editdistance, sys

from pipeline.regions_search import get_similar_xpaths

THRESHOLD = 1

def similar(baseline, xpaths):
  result = []

  for xpath in xpaths:
    arr1 = baseline.split('/')
    arr2 = xpath.split('/')

    while len(arr1) < len(arr2): arr1.append('')
    while len(arr2) < len(arr1): arr2.append('')

    if editdistance.eval(arr1, arr2) <= THRESHOLD: result.append(True)
    else: result.append(False)

  return result


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
