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
          })


  result = pd.DataFrame([])
  for cluster in clusters:
      result = pd.concat([result, pd.DataFrame([cluster['higher_probability']])])

  return result
