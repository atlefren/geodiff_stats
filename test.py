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


async def get_column(table, column):
  conn = await get_conn()
  try:
    res = await conn.fetch(f'SELECT (data->>\'{column}\')::float from {table} where data->\'CreateError\' is null')
    return np.array([r[0] for r in res])
  finally:
    await conn.close()


async def get_check_column(table, column):
  conn = await get_conn()
  try:
    res = await conn.fetch(f'SELECT (data->>\'{column}\')::boolean from {table} where data->\'CreateError\' is null')
    return np.array([r[0] for r in res])
  finally:
    await conn.close()

async def get_counts(table):
  conn = await get_conn()
  try:
    res = await conn.fetch(f'SELECT id::bigint, numpoints from {table}')

    return pd.DataFrame({
      'id': np.array([r[0] for r in res]),
      'numpoints': np.array([r[1] for r in res]).astype(np.float),
    })

  finally:
    await conn.close()


async def get_columns(table):
  conn = await get_conn()
  try:
    res = await conn.fetch(f'''
      SELECT 
        id,
        (data->>\'CreateTime\')::float,
         (data->>\'ApplyTime\')::float,
         (data->>\'UndoTime\')::float,
         (data->>\'PatchSize\')::float,
         (data->>\'ForwardCorrect\')::boolean,
         (data->>\'UndoCorrect\')::boolean,
         (data->>\'CreateError\')::text
      FROM 
        {table} 
      --WHERE
        --data->\'CreateError\' is null
      
      ''')

    cols = {
      'id': [],
      'CreateTime': [],
      'ApplyTime': [],
      'UndoTime': [],
      'PatchSize': [],
      'ForwardCorrect': [],
      'UndoCorrect': [],
      'CreateError': []
    }
    for r in res:
      cols['id'].append(r[0])
      cols['CreateTime'].append(r[1])
      cols['ApplyTime'].append(r[2])
      cols['UndoTime'].append(r[3])
      cols['PatchSize'].append(r[4])
      cols['ForwardCorrect'].append(r[5])
      cols['UndoCorrect'].append(r[6])
      cols['CreateError'].append(r[7])

    return {
      'id': np.array(cols['id']).astype(np.int64),
      'CreateTime': np.array(cols['CreateTime']),
      'ApplyTime': np.array(cols['ApplyTime']),
      'UndoTime': np.array(cols['UndoTime']),
      'PatchSize': np.array(cols['PatchSize']),
      'ForwardCorrect': np.array(cols['ForwardCorrect']),
      'UndoCorrect': np.array(cols['UndoCorrect']),
      'CreateError': np.array(cols['CreateError']),
    }

  finally:
    await conn.close()


async def count_correct(table, type, column):
  conn = await get_conn()
  try:
    return await conn.fetchval(f'SELECT SUM(CASE WHEN ({type}->>\'{column}\')::boolean = true THEN 1 ELSE 0 END)::DECIMAL from {table};')    
  finally:
    await conn.close()




async def get_dataframe(geom_type, differs, metrics, checks):
  frames = []
  for differ in differs:
    table = get_table_name(differ, geom_type)
    cols = await get_columns(table)

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


def get_table_name(differ, geom_type):
  return f'results.{differ}_{geom_type}'


async def add_counts(geom_type, df, count_tables):
  if geom_type in count_tables:
      counts = await get_counts(count_tables[geom_type])
      df = pd.merge(df, counts, on='id')
      df['quantile_10'] = pd.cut(df['numpoints'], bins=10, precision=0)
  return df


def stats_tests(df, metrics, groups, group_by):

  p_value = 0.05

  results = {}

  for metric in metrics:
      #print(f'Testing {metric}\n')

      #print('Testing normal distribution:')

      normality = {}
      for differ, ids in groups.items():
        normal = is_normal(df.loc[ids, metric].values, p_value)
        normality[differ] = normal
        #print(f'{differ} \t: {normal}')

      # get an array pr differ for selected metric
      data_for_metric = [df.loc[ids, metric].values for ids in groups.values()]
      #h0: samples from same distribution
      from_same_distribution = kruskal(data_for_metric, p_value)

      #print(f'\nIs from same distribution: {from_same_distribution}')

      #print('Post-hoc test (conovers test)')
      conover_result = sp.posthoc_conover(df, val_col=metric, group_col=group_by, p_adjust='holm')

      conover_arr = []
      for m1 in groups:
        for m2 in groups:
          if m1 != m2:
            res = conover_result.loc[m1, m2] 
            conover_arr.append(res < p_value)

      results[metric] = {
        'normality': normality,
        'from_same_distribution': from_same_distribution,
        'post_hoc': conover_result if False in conover_arr else None,
      }
  return results


def get_normal(data):
  res = []
  for key, val in data.items():
    for k, v in val['normality'].items():
      if v == True:
        res.append(f'{key} - {k}')
  return res;


def get_from_same_dist(data):
  res = []
  for key, val in data.items():
    if val['from_same_distribution']:
      res.append(key)
  return res


def get_post_hoc(data):
  res = []
  for key, val in data.items():
    if val['post_hoc']:
      res.append(key)
  return res


def calculate_stats(df, metrics, group_by, differs):

    #print(df)

    

    print('differ\tNum\tCreateError\tPatchError\tUndoError')
    for differ in differs:
      rows = df[df["differ"]==differ]
      failed_create = rows[rows['CreateError'].notnull()]
      failed_apply = rows[rows['ForwardCorrect'] == False]
      failed_undo = rows[rows['UndoCorrect'] == False]

      print(f'{differ}\t{len(rows)}\t{len(failed_create)}\t\t{len(failed_apply)}\t\t{len(failed_undo)}')

    df_ok = df# df[(df['CreateError'].isnull())]

  

    df_ok = df_ok.assign(CreateTime=np.where(df_ok.CreateError.notnull(), float('NaN'), df_ok.CreateTime))
    df_ok = df_ok.assign(ApplyTime=np.where(df_ok['ForwardCorrect'] == False, float('NaN'), df_ok.ApplyTime))
    df_ok = df_ok.assign(UndoTime=np.where(df_ok['UndoCorrect'] == False, float('NaN'), df_ok.UndoTime))
    df_ok = df_ok.assign(PatchSize=np.where(df_ok['ForwardCorrect'] == False, float('NaN'), df_ok.PatchSize))

    grouped = df_ok.groupby(group_by)
    groups = grouped.groups


    print('\n')
    print('Count')
    print(grouped[metrics].count())


    print('\n')
    print('Mean')
    print(grouped[metrics].mean().round(0))
    print('\n')
    print('Std.dev')
    print(grouped[metrics].std().round(0))
    print('\n')

    return

    res = stats_tests(df, metrics, groups, group_by)
    normal_distributions = get_normal(res) 
    print(f'Normal distributions: {len(normal_distributions)}')
    if len(normal_distributions) > 0:
       print(f'Normal distributions: {normal_distributions}')
    
    from_same_dist = get_from_same_dist(res) 
    print(f'From same distribution: {len(from_same_dist)}')
    if len(from_same_dist) > 0:
       print(f'From same distribution: {from_same_dist}')
    
    post_hoc_data = get_post_hoc(res)
    print(f'Failed post hoc test: {len(post_hoc_data)}')


async def main():
  
  differs = ['text', 'geojson', 'binary', 'geom']
  #geom_types = ['point2', 'linestring', 'polygon']
  geom_types = ['linestring']

  metrics = ['CreateTime', 'ApplyTime', 'UndoTime', 'PatchSize']
  checks = ['ForwardCorrect', 'UndoCorrect']
  group_by = 'differ'

  count_tables = {
    'linestring': 'osm.line_numpoints',
    'polygon': 'osm.polygon_numpoints'
  }

  for geom_type in geom_types:
    print(f'# Results for {geom_type}')

    df = await get_dataframe(geom_type, differs, metrics, checks)
    df = await add_counts(geom_type, df, count_tables)

    #calculate_stats(df, metrics, group_by)
    
    calculate_stats(df, metrics, group_by, differs)

    a = df.groupby('quantile_10')
    #print(a.groups)
    #for quantile, df in a:
    #  print("!!", quantile)
    #  calculate_stats(df, metrics, group_by)


asyncio.get_event_loop().run_until_complete(main())



'''
    print('Num correct')
    for differ in differs:
      print(differ)
      for check in checks:
        n = await count_correct(table, differ, check)
        print(f'\t{check}\t: {n}')
    '''