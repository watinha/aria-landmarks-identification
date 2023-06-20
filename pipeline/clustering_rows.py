import pandas as pd

CHUNKSIZE = 50000

landmarks = ['banner', 'complementary', 'contentinfo', 'form',
             'navigation', 'search', 'region', 'main']
def cluster_rows ():

    for landmark in landmarks:
        nrows = CHUNKSIZE
        with pd.read_csv('./results/test/classified-%s.csv' % (landmark), iterator=True) as parser:
            chunk_count = 0
            while nrows == CHUNKSIZE:
                chunk_count = chunk_count + 1
                df = parser.get_chunk(CHUNKSIZE)
                nrows, _ = df.shape
                urls = pd.unique(df['url']).tolist()

                for url_ind, url in enumerate(urls):
                    df_url = df.loc[df['url'] == url, :]

                    result = cluster_landmarks(df_url, landmark)

                    if (result.size > 0):
                        result.to_csv('./results/clusters/%s-%d-%d-results.csv' % (landmark, url_ind, chunk_count))


def cluster_landmarks (df_url, landmark):
  clusters = []
  nrows_url, _ = df_url.shape

  print('%s' % (nrows_url))
  print(' - searching for clusters')
  for i in range(nrows_url-1, -1, -1):
      target = df_url.iloc[i, :]
      left, right, top, bottom = target['left'], target['left'] + target['width'], target['top'], target['top'] + target['height']
      count = 0

      for i, cluster in enumerate(clusters):
          if (cluster['higher_probability']['xpath'].startswith(target['xpath'])):
              if cluster['higher_probability'][landmark] < target[landmark]:
                  cluster['higher_probability'] = target

                  print(' - merging clusters')
                  to_remove = [] # merge clusters
                  for j in range(i+1, len(clusters)):
                      cluster_2 = clusters[j]
                      if (cluster['higher_probability']['xpath'].startswith(cluster_2['higher_probability']['xpath']) or
                          cluster_2['higher_probability']['xpath'].startswith(cluster['higher_probability']['xpath'])):
                          if (cluster['higher_probability'][landmark] < cluster_2['higher_probability'][landmark]):
                              clusters[i]['higher_probability'] = cluster_2['higher_probability']
                              clusters[i]['left'] = cluster_2['left']
                              clusters[i]['right'] = cluster_2['right']
                              clusters[i]['top'] = cluster_2['top']
                              clusters[i]['bottom'] = cluster_2['bottom']

                          to_remove.append(j)

                  print(' - %s' % (to_remove))
                  to_remove.reverse()
                  for remove_index in to_remove:
                       clusters.pop(remove_index)

              break
          else:
              count = count + 1

      if count == len(clusters):
          clusters.append({
              'higher_probability': target,
              'left': left,
              'right': right,
              'top': top,
              'bottom': bottom,
          })

  print(' - removing overflown clusters')
  to_remove = [] # removing overflown landmarks
  for i in range(len(clusters)):
    cluster_1 = clusters[i]
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
  print(to_remove)
  for remove_index in to_remove:
    if remove_index in clusters:
      clusters.pop(remove_index)


  result = pd.DataFrame([])
  for cluster in clusters:
      result = pd.concat([result, pd.DataFrame([cluster['higher_probability']])])

  return result
