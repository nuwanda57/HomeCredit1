import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

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