import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def create_submission(ans):
    test = pd.read_csv('data/application_test.csv')
    test_ids = test['SK_ID_CURR']
    submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': ans})
    return submission

def draw_pies_for_columns(columns, dff):
    for col in columns:
        counts = Counter(dff[col])
        df = pd.DataFrame.from_dict(counts, orient='index')
        df.index.name = col
        df.columns = ['count']
        df.plot(kind='pie', rot=0, figsize=(5, 5), y='count', autopct='%1.1f%%')
        plt.title('Data distribution by %s' % col)
        plt.xlabel(col)
        plt.ylabel('count')
        plt.show()
        
def draw_hists_for_categorical_columns(columns, dff):
    for col in columns:
        counts = Counter(dff[col])
        df = pd.DataFrame.from_dict(counts, orient='index')
        df.index.name = col
        df.columns = ['count']
        df.plot(kind='bar', rot=90, figsize=(15,4))
        plt.title('Data distribution by %s' % col)
        plt.xlabel(col)
        plt.ylabel('count')
        plt.show()
        
def draw_hists_for_numerical_columns(columns, dff, quantile=None, bins=50):
    for col in columns:
        plt.figure(figsize=(16,6))
        plt.hist(dff[col], bins=bins)
        plt.title('Data distribution by %s' % col)
        plt.xlabel(col)
        plt.show()
        if quantile is None:
            continue
        plt.figure(figsize=(16,6))
        plt.hist(dff[col][dff[col] < dff[col].quantile(quantile)], bins=bins)
        plt.title('Data distribution by %s, quantile %s' % (col, quantile))
        plt.xlabel(col)
        plt.show() 

def draw_correlations_for_numerical_columns(columns, dff):
    plt.figure(figsize=(18, 18))
    cor = dff[columns].corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    plt.show()

def get_df_general_stats(df):
    column_statistics = pd.DataFrame(index=df.columns)
    column_statistics['CountNaN'] = df.isna().sum()
    column_statistics['RateNaN%'] = (df.isna().sum() / df.shape[0] * 100).astype(int)
    column_statistics['CountUnique'] = df.nunique()
    described = df.describe()
    column_statistics['MinValue'] = described.loc['min']
    column_statistics['MaxValue'] = described.loc['max']
    column_statistics['Mean'] = described.loc['mean']
    column_statistics['Std'] = described.loc['std']
    column_statistics['Median'] = described.loc['50%']
    return column_statistics

def get_nan_cols_stats(column_statistics):
    print('Число признаков, у которых в данных есть пропуски: {} из {}'.format(
            column_statistics[column_statistics['CountNaN'] != 0].shape[0], 
            column_statistics.shape[0]))

    return column_statistics[column_statistics['CountNaN'] != 0].sort_values(
            by='RateNaN%', ascending=False)


def woe(df):
    target = df['TARGET']
    columns = df.columns
    woe_table = pd.DataFrame(columns=[
        'woe', 'iv', '%target=1', '%target=0', 'min', 'max', 'count', '%_of_data'])
    main_woe_table = pd.DataFrame(columns=[
        'woe', 'iv', '%target=1', '%target=0', 'min', 'max', 'count', '%_of_data'])
    description = df.describe()
    count_1 = df[df.TARGET == 1]['TARGET'].count()
    count_0 = df[df.TARGET == 0]['TARGET'].count()
    for col in columns:
        b1 = description.loc['25%', col]
        b2 = description.loc['50%', col]
        b3 = description.loc['75%', col]
        iv_total = 0
        old_left = None
        values = np.unique(df[col])
        if len(values) <= 20:
            for v in values:
                tmp = df[df[col] == v]
                count = tmp[col].count()
#                 if count / (count_1 + count_0) < 0.05:
#                     print('too little data: for col={}, value={}: {}'.format(
#                             col, v, count / (count_1 + count_0)))
                percent_of_data = count / df.shape[0] * 100
                percent_of_one = tmp[tmp.TARGET == 1][col].count() / count_1 * 100
                percent_of_zero = tmp[tmp.TARGET == 0][col].count() / count_0 * 100
                woe = 0
                if percent_of_zero != 0:
                    woe = np.log(percent_of_one / percent_of_zero)
                iv = (percent_of_one - percent_of_zero) * woe
                woe_table.loc[col + '__{}'.format(v)] = {
                    'woe': woe, 'iv': iv, '%target=1': percent_of_one, '%target=0': percent_of_zero,
                    'min': v, 'max': v, 'count': count, '%_of_data': percent_of_data
                }
                iv_total += iv
        else:
            for b in [(df[col].min(), b1), (b1, b2), (b2, b3), (b3, df[col].max())]:
                left, right = b
                if old_left == left:
                    continue
                old_left = left
                tmp = df[(df[col] >= left) & (df[col] < right)]
                if left == df[col].min():
                    tmp = df[df[col] < right]
                elif right == df[col].max():
                    tmp = df[df[col] >= left]
                count = tmp[col].count()
                tmp_min = tmp[col].min()
                tmp_max = tmp[col].max()
                percent_of_data = count / df.shape[0] * 100
                percent_of_one = tmp[tmp.TARGET == 1][col].count() / count_1 * 100
                percent_of_zero = tmp[tmp.TARGET == 0][col].count() / count_0 * 100
                woe = 0
                if percent_of_zero != 0:
                    woe = np.log(percent_of_one / percent_of_zero)
                iv = (percent_of_one - percent_of_zero) * woe
                woe_table.loc[col + '__%s_%s' % (left, right)] = {
                    'woe': woe, 'iv': iv, '%target=1': percent_of_one, '%target=0': percent_of_zero,
                    'min': tmp_min, 'max': tmp_max, 'count': count, '%_of_data': percent_of_data
                }
                iv_total += iv
        tmp = df
        count = tmp[col].count()
        tmp_min = tmp[col].min()
        tmp_max = tmp[col].max()
        iv = iv_total
        main_woe_table.loc[col] = {
            'woe': None, 'iv': iv, '%target=1': None, '%target=0': None,
            'min': tmp_min, 'max': tmp_max, 'count': count, '%_of_data': None
        }   
    return main_woe_table, woe_table

