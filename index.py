import asyncio
import asyncpg
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import scipy.stats as ss
from scipy.stats import normaltest

import numpy as np 
from dotenv import load_dotenv


load_dotenv()



async def get_conn():
  return await asyncpg.connect(os.environ['CONN_STR'])


async def get_column(table, type, column):
  conn = await get_conn()
  try:
    res = await conn.fetch(f'SELECT ({type}->>\'{column}\')::float from {table} LIMIT 10;')
    return np.array([r[0] for r in res])
  finally:
    await conn.close()

'''
def is_gaussian(data):
  stat, p = normaltest(data)
  alpha = 0.05
  return p > alpha


def are_equal(data):
  stat, p = friedmanchisquare(*data)
  alpha = 0.05
  return p > alpha
'''


async def get_dataframe(table, differs, metrics):
  frames = []
  for differ in differs:
    cols = {}
    for metric in metrics:
      data = await get_column('osm.point_results', differ, metric)
      cols[metric] = data
    frame = pd.DataFrame(cols)
    frame['differ'] = differ
    frames.append(frame)
    #frames[metric] = pd.DataFrame(d)
  return pd.concat(frames)


async def get_data(table, differs, metrics):
  m = {}
  for metric in metrics:
    m[metric] = {}
    for differ in differs:
      data = await get_column('osm.point_results', differ, metric)
      m[metric][differ] = data
  return m


def is_normal(values, alpha):
  stat, p = normaltest(values)
  return p > alpha


def test(data, metric):
  print(f'\n\ntest {metric}\n')
  differs = data[metric]
  for differ, values in differs.items():
      mean = np.mean(values)
      std = np.std(values)
      print('%s\t%.4f\t%.4f\t%s' % (differ, mean, std, is_normal(values, 0.05)))
  
  
  
  H, p = ss.kruskal(*differs.values())
  H2, p2 = ss.friedmanchisquare(*differs.values())
  print('Friedman test ', p2, p2 < 0.05)
  print('Kruskal-Wallis test', p, p < 0.05)


def test_df(df, metrics):
  print(df.columns)
  print(df.head())
  
  data = [df.loc[ids, 'CreateTime'].values for ids in df.groupby('differ').groups.values()]
  print(data)
  H, p = ss.kruskal(*data)
  print(p)

  '''
  for metric in metrics:
    data = [df.loc[ids, metric].values for ids in df.groupby('differ').groups.values()]
    print(metric)
    H, p = ss.kruskal(*data)
    print(p)
  '''

  '''
  #grouped = df.groupby('differ')

  #print(grouped.mean())
  #print(grouped.std())

  #for metric in metrics:
  #  #g = df.loc[df, metric]
  #  print(f'\n\ntest {metric}\n')
  #  df.loc[df['column_name'] == some_value]
  '''
    
    
    
  '''
    #H, p = ss.kruskal(*data)
    #print('Kruskal-Wallis test', p, p < 0.05)
  #for differ, df in differs.items():
      #mean = np.mean(values)
      #std = np.std(values)
      #print('%s\t%.4f\t%.4f\t%s' % (differ, mean, std, is_normal(values, 0.05)))
  #H, p = ss.kruskal(*differs.values())
  #H2, p2 = ss.friedmanchisquare(*differs.values())
  #print('Friedman test ', p2, p2 < 0.05)
  #print('Kruskal-Wallis test', p, p < 0.05)
  '''

async def main():
  differs = ['textdiffer', 'jsondiffer',  'geomdiffer', 'binarydiffer'] # 
  metrics = ['CreateTime', 'ApplyTime', 'UndoTime']

  
  data = await get_data('osm_test.point_results', differs, metrics)
  
  for metric in ['CreateTime']:
    test(data, metric)
  

  df = await get_dataframe('osm_test.point_results', differs, metrics)
  
  test_df(df, metrics)

  '''
  for metric, differs in data.items():
    print(metric)
    for differ, values in differs.items():
      stat, p = normaltest(values)
      print('\t',differ, np.mean(values), np.std(values), p)
    H2, p2 = ss.friedmanchisquare(*differs.values())
    H, p = ss.kruskal(*differs.values())
    print('Kruskal-Wallis test', metric, p, p < 0.05)
    print('Friedman test ', metric, p2, p2 < 0.05)
  '''

  '''
  df = await get_dataframe('osm_test.point_results', differs, metrics)
  grouped = df.groupby('algorithm')

  print('mean')
  print(grouped.mean())
  
  print('stddev')
  print(grouped.std())

  
  for metric in metrics:
    for differ in differs:
      ax = df.loc[df['algorithm'] == differ][metric].plot.hist(title=f'{metric} - {differ}')
      fig = ax.get_figure()
      fig.savefig(f'{metric}_{differ}.png')
      plt.close()

  for metric in metrics:
    data = [df.loc[ids, metric].values for ids in df.groupby('algorithm').groups.values()]
    H, p = ss.kruskal(*data)
    print('Kruskal-Wallis test', metric, p, p < 0.05)
    H2, p2 = ss.friedmanchisquare(*data)
    print('Friedman test ', metric, p2, p2 < 0.05)
  '''

asyncio.get_event_loop().run_until_complete(main())