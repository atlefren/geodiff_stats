import os
import asyncio
import asyncpg
import numpy as np
import pandas as pd 
import scipy.stats as ss
import scikit_posthocs as sp
from dotenv import load_dotenv

# https://github.com/maximtrp/scikit-posthocs
# 
# time: ms
# size: bytes
#


load_dotenv()

async def get_conn():
  return await asyncpg.connect(os.environ['CONN_STR'])


async def get_column(table, type, column):
  conn = await get_conn()
  try:
    res = await conn.fetch(f'SELECT ({type}->>\'{column}\')::float from {table};')
    return np.array([r[0] for r in res])
  finally:
    await conn.close()


async def count_correct(table, type, column):
  conn = await get_conn()
  try:
    return await conn.fetchval(f'SELECT SUM(CASE WHEN ({type}->>\'{column}\')::boolean = true THEN 1 ELSE 0 END)::DECIMAL from {table};')    
  finally:
    await conn.close()




async def get_dataframe(table, differs, metrics):
  frames = []
  for differ in differs:
    cols = {}
    for metric in metrics:
      data = await get_column(table, differ, metric)
      cols[metric] = data
    frame = pd.DataFrame(cols)
    frame['differ'] = differ
    frames.append(frame)
    #frames[metric] = pd.DataFrame(d)
  df = pd.concat(frames)
  df.index = [x for x in range(1, len(df.values) + 1)]
  return df


def is_normal(values, alpha):
  stat, p = ss.normaltest(values)
  return p > alpha


def kruskal(values, alpha):
  H, p = ss.kruskal(*values)
  return p > alpha


async def main():
  #tables = ['osm.point_results', 'osm.linestring_results', 'osm.polygon_results']
  tables = ['osm_test.point_results', 'osm_test.line_results', 'osm_test.polygon_results']
  differs = ['textdiffer', 'jsondiffer',  'geomdiffer', 'binarydiffer'] # 
  metrics = ['CreateTime', 'ApplyTime', 'UndoTime', 'PatchSize']

  checks = ['ForwardCorrect', 'UndoCorrect']

  group_by = 'differ'

  for table in tables:
    print(f'# Results for {table}')

    df = await get_dataframe(table, differs, metrics)
    grouped = df.groupby(group_by)
    groups = grouped.groups



    print('\n')
    print('Count')
    print(grouped.count())
    print('\n')
    print('Mean')
    print(grouped.mean())
    print('\n')
    print('Std.dev')
    print(grouped.std())
    print('\n')

    print('Num correct')
    for differ in differs:
      print(differ)
      for check in checks:
        n = await count_correct(table, differ, check)
        print(f'\t{check}\t: {n}')

    for metric in metrics:
      print(f'Testing {metric}\n')

      print('Testing normal distribution:')
      for differ, ids in groups.items():
        print(f'{differ} \t: {is_normal(df.loc[ids, metric].values, 0.05)}')
      
      # get an array pr differ for selected metric
      data_for_metric = [df.loc[ids, metric].values for ids in groups.values()]
      #h0: samples from same distribution
      from_same_distribution = kruskal(data_for_metric, 0.05)

      print(f'\nIs from same distribution: {from_same_distribution}')
      

      print('Post-hoc test (conovers test)')
      print(sp.posthoc_conover(df, val_col=metric, group_col=group_by, p_adjust='holm'))

      print('-----')


asyncio.get_event_loop().run_until_complete(main())
