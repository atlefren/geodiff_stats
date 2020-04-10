import os
import asyncio
import asyncpg
import numpy as np
import pandas as pd 
import scipy.stats as ss
import scikit_posthocs as sp
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from dotenv import load_dotenv

pd.set_option('mode.chained_assignment', None)
load_dotenv()


DIFFER_NAMES = {
    'text': 'TextDiff',
    'geojson': 'JsonDiff',
    'binary': 'BinaryDiff',
    'geom': 'GeomDiff'
}

GEOMETRY_NAMES = {
    'point2': 'Point',
    'linestring': 'Linestring',
    'polygon': 'Polygon'
}


async def get_conn():
  return await asyncpg.connect(os.environ['CONN_STR'])


async def get_numpoints(table):
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

'''
async def count_correct(table, type, column):
  conn = await get_conn()
  try:
    return await conn.fetchval(f'SELECT SUM(CASE WHEN ({type}->>\'{column}\')::boolean = true THEN 1 ELSE 0 END)::DECIMAL from {table};')    
  finally:
    await conn.close()
'''



async def get_dataframe(geom_type, differs, checks):
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


async def add_counts(geom_type, df, numpoints_tables):
  if geom_type in numpoints_tables:
    numpoints = await get_numpoints(numpoints_tables[geom_type])
    df = pd.merge(df, numpoints, on='id')
  return df


def stats_tests(df, metrics, groups, group_by):
  p_value = 0.05
  results = {}
  for metric in metrics:
      normality = {}
      for differ, ids in groups.items():
        normal = is_normal(df.loc[ids, metric].values, p_value)
        normality[differ] = normal

      # get an array pr differ for selected metric
      data_for_metric = [df.loc[ids, metric].values for ids in groups.values()]
      #h0: samples from same distribution
      from_same_distribution = kruskal(data_for_metric, p_value)

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
    if val['post_hoc'] is not None:
      res.append(key)
  return res


async def main():

    checks = ['ForwardCorrect', 'UndoCorrect']

    numpoints_tables = {
        'linestring': 'osm.line_numpoints',
        'polygon': 'osm.polygon_numpoints'
    }

    differs = ['text', 'geojson', 'binary', 'geom']
    geom_types = ['point2', 'linestring', 'polygon']

    functions = [
        #print_num_observations,
        #calculate_stats,
        #record_errors,
        geom_type_filter(compute_average_numpoints, ['linestring', 'polygon']),
        #geom_type_filter(with_filter(createtime_top_percentile_removed, remove_errors), ['linestring', 'polygon']),
        #geom_type_filter(with_filter(createtime_compute_correlations, remove_errors),['linestring', 'polygon']),
        geom_type_filter(with_filter(create_createtime_vs_numpoints_barchart, remove_errors), ['linestring', 'polygon'])
    ]

    for geom_type in geom_types:
        print(f'\n# Results for {GEOMETRY_NAMES[geom_type]}\n-----------------')
        df = await load_dataframe(geom_type, differs, checks, numpoints_tables)
        for f in functions:
            f(geom_type, df, differs)



async def load(geom_type, differs, checks, numpoints_table):
    df = await get_dataframe(geom_type, differs, checks)
    df = await add_counts(geom_type, df, numpoints_tables)
    return df


def calculate_stats(geom_type, df, differs):
    group_by = 'differ'
    metrics = ['CreateTime', 'ApplyTime', 'UndoTime', 'PatchSize']

    #remove completely those with creation error
    df_ok = remove_errors(df)

    # use nans for apply/undo errors
    df_ok = df_ok.assign(ApplyTime=np.where(df_ok['ForwardCorrect'] == False, float('NaN'), df_ok.ApplyTime))
    df_ok = df_ok.assign(UndoTime=np.where(df_ok['UndoCorrect'] == False, float('NaN'), df_ok.UndoTime))
    df_ok = df_ok.assign(PatchSize=np.where(df_ok['ForwardCorrect'] == False, float('NaN'), df_ok.PatchSize))

    grouped = df_ok.groupby(group_by)
    groups = grouped.groups

    print('\nResults\n')

    means = grouped[metrics].mean()
    stds = grouped[metrics].std()

    l1 = [f'{"Differ":15}']
    l2 = [f'{"":15}']
    for metric in metrics:
        l1.append(f'{metric:20}\t\t')
        l2.append(f'{"Mean":10}\t{"St.dev":10}\t')

    print(''.join(l1))
    print(''.join(l2))
    for differ in differs:
        line = [f'{DIFFER_NAMES[differ]:15}']
        for metric in metrics:
            if metric == 'PatchSize':
                line += [f'{means[metric].loc[differ]:10,.0f}\t', f'{stds[metric].loc[differ]:10,.0f}\t']
            else:
                line += [f'{means[metric].loc[differ]:10,.2f}\t', f'{stds[metric].loc[differ]:10,.2f}\t']

        print(''.join(line))

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
    if len(post_hoc_data) > 0:
      print(f'Failed post hoc test: {post_hoc_data}')
    

def print_num_observations(geom_type, df, differs):
    print('Num observations')
    for differ, rows in split_by_differ(df, differs):
        print(f'{DIFFER_NAMES[differ]:10}: {len(rows):,}')

def remove_errors(df):
    return df[(df['CreateError'].isnull())]


def geom_type_filter(f, types):
    def func(geom_type, df, differs):
        if geom_type in types:
            return f(geom_type, df, differs)
    return func


def with_filter(f, filter):
    def func(geom_type, df, differs):
        return f(geom_type, filter(df), differs)
    return func


async def load_dataframe(geom_type, differs, checks, numpoints_tables):
    pickle_file = f'./{geom_type}.pkl'

    if os.path.isfile(pickle_file):
      return pd.read_pickle(pickle_file)
    else:
      df = await load(geom_type, differs, checks, numpoints_tables)
      df.to_pickle(pickle_file)
      return df


def get_remove_top_percentile(metric, quantile=0.99):
    def rem(df):
        quantile_val = df[metric].quantile(quantile)
        return df[df[metric] <= quantile_val]
    return rem


def get_get_top_percentile(metric, quantile=0.99):
    def get(df):
        quantile_val = df[metric].quantile(0.99)
        return df[df[metric] > quantile_val]
    return get


def split_by_differ(df, differs, includeidx=False):
    for i, differ in enumerate(differs):
        if includeidx:
            yield differ, df[df["differ"]==differ], i
        else:
            yield differ, df[df["differ"]==differ]


def compute_stats(title, df, metric, differs, filter=None):

    print(f'\n{title}\n')
    print(f'{"Differ":10}\t{"Mean":10}\t{"St.dev":10}')
    for differ, rows in split_by_differ(df, differs):
        filtered = rows if filter is None else filter(rows)

        std = filtered[metric].std().round(2)
        name = DIFFER_NAMES[differ]
        print(f'{name:10}\t{get_mean(filtered, metric):10,.2f}\t{std:10,.2f}')


def get_mean(df, metric):
    return df[metric].mean()


def record_errors(geom_type, df, differs):


    lines = [f'{"Differ":10}\t{"Create":10}\t{"Apply":10}\t{"Undo":10}']
    percentage_lines = [f'{"Differ":10}\t{"Create":10}\t{"Apply":10}\t{"Undo":10}']
    create_errors = []
    for differ, rows in split_by_differ(df, differs):
        count_all = len(rows)
        count_created = len(remove_errors(rows))
        failed_create = rows[rows['CreateError'].notnull()]
        count_failed_create = len(failed_create)
        failed_apply = len(rows[rows['ForwardCorrect'] == False])
        failed_undo = len(rows[rows['UndoCorrect'] == False])
        
        failed_create_percentage = count_failed_create / count_all * 100
        failed_apply_percentage = failed_apply / count_created * 100
        failed_undo_percentage = failed_undo / count_created * 100
        name = DIFFER_NAMES[differ]
        lines.append(f'{name:10}\t{count_failed_create:10,.0f}\t{failed_apply:10,.0f}\t{failed_undo:10,.0f}')
        percentage_lines.append(f'{name:10}\t{failed_create_percentage:10,.0f}%\t{failed_apply_percentage:10,.0f}%\t{failed_undo_percentage:10,.0f}%')

        u = failed_create['CreateError'].unique()
        create_errors.append(f'{name:10}: {", ".join(u) if len(u) > 0 else "-"}')


    print(f'\n{"Errors"}\n')
    print("\n".join(lines))

    print(f'\n{"% Errors"}\n')
    print("\n".join(percentage_lines))

    print(f'\n{"Create error messages"}\n')

    print('\n'.join(create_errors))

    print('\n\n')

    for differ, rows in split_by_differ(df, differs):
      pass


def createtime_compute_correlations(geom_type, df, differs):

    metric = 'CreateTime'
    numpoints = 'numpoints'

    title = f'Correlation {metric} - {numpoints}'

    get_top_percentile = get_get_top_percentile(metric)

    print(f'\n{title}\n')

    print(f'{"Differ":10}\t{"All":10}\t{"Top 1%":10}')
    for differ, rows in split_by_differ(df, differs):
        filtered_rows = get_top_percentile(rows)

        all = rows[metric].corr(rows[numpoints])
        top = filtered_rows[metric].corr(filtered_rows[numpoints])
        name = DIFFER_NAMES[differ]
        print(f'{name:10}\t{all:10,.2f}\t{top:10,.2f}')




def compute_average_numpoints(geom_type, df, differs):
    numpoints = 'numpoints'
    avg_all = df[numpoints].mean().round(0)
    print(f'Average numpoints = {avg_all}')

    print("Numpoints in failed")
    for differ, rows in split_by_differ(df, differs):
        failed_create = rows[rows['CreateError'].notnull()]

        print(f'{differ:10} = {get_mean(failed_create, numpoints):10,.0f}')


def createtime_top_percentile_removed(geom_type, df, differs):
    metric = 'CreateTime'
    compute_stats('CreateTime with upper 99 percentile excluded', df, metric, differs, get_remove_top_percentile(metric))


def autolabel(rects, ax, max):
    for rect in rects:
        value = rect.get_height()
        if value/max < 0.02:
            ax.annotate(f"{value:,.2f}",
                        xy=(rect.get_x() + rect.get_width() / 2, value),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')


def plot_bar(ax, df, differ, labels):

    grouped = df.groupby('group') \
       .agg(count=('group', 'size'), mean=('CreateTime', 'mean')) \
       .reset_index()

    print(DIFFER_NAMES[differ])
    print(grouped)

    rects = ax.bar(grouped['group'], grouped['mean'], align='edge', width=0.3)
    max = grouped['mean'].max()
    autolabel(rects, ax, max)
    ax.set_ylim([0, max * 1.1])
    ax.set_title(DIFFER_NAMES[differ])
    ax.set_xticklabels(labels, rotation=45)
    ax.set_xticks(grouped['group'])
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))


def get_labels(bins):
    return [f'({bins[i]:,}–{bins[i + 1]:,}]' for i in range(len(bins) - 1)]


def create_createtime_vs_numpoints_barchart(geom_type, df, differs):

    bins = [0, 20, 40, 60, 80, 100, 150, 200, 300, 500, 1000,  3000]
    labels = get_labels(bins)
    df['group'] = pd.cut(df['numpoints'], bins=bins, labels=labels)
    if False:
        df['group'], bins = pd.qcut(df['numpoints'], 15, duplicates="drop", labels=False, retbins=True)
        labels = get_labels(bins)

    print(f'\n{"CreateTime vs. numpoints barchart"}\n')

    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.subplots_adjust(hspace=0.35, wspace=0.15)
    x = [0, 0, 1, 1]
    y = [0, 1, 0, 1]

    for differ, rows, i in split_by_differ(df, differs, includeidx=True):
        plot_bar(axs[x[i], y[i]], rows, differ, labels)

    y_label = 'CreateTime (ms)'
    x_label = 'Number of vertices'
    axs[0, 0].set(ylabel=y_label)
    axs[1, 0].set(ylabel=y_label)
    axs[1, 0].set(xlabel=x_label)
    axs[1, 1].set(xlabel=x_label)

    fig.suptitle(GEOMETRY_NAMES[geom_type], fontsize=16)
    filename = f'barchart_createtime_vs_numpoints_{geom_type}.png'
    fig.savefig(filename)
    print(f'Saved {filename}')


asyncio.get_event_loop().run_until_complete(main())