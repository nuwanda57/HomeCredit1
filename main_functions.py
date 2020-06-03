import pandas as pd
import numpy as np
from copy import deepcopy

import lightgbm as lgb
from sklearn.model_selection import train_test_split
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import helpers

def add_renamed(feature, application_df, resultive_df,
                application_features_numeric, application_features_categorical):
    assert resultive_df[resultive_df[feature].isna()].shape[0] == 0
    application_features_numeric.append('application__{}'.format(feature))
    resultive_df['application__{}'.format(feature)] = resultive_df[feature]
    resultive_df.drop([feature], axis=1, inplace=True)

def add_renamed_nan(feature, application_df, resultive_df,
                    application_features_numeric, application_features_categorical):
    application_features_numeric.append('application__{}'.format(feature))
    resultive_df['application__{}'.format(feature)] = resultive_df[feature]
    resultive_df.drop([feature], axis=1, inplace=True)

def add_quantile_nan(feature, quantile_dict, application_df, resultive_df, 
                     application_features_numeric, application_features_categorical):
    application_features_numeric.append('application__{}'.format(feature))
    resultive_df['application__{}'.format(feature)] = resultive_df[feature]
    if quantile_dict is None:
        ql, qr = resultive_df[feature].quantile(0.01), resultive_df[feature].quantile(0.99)
    else:
        ql, qr = quantile_dict[feature]
    resultive_df.loc[resultive_df[feature] > qr, feature] = qr
    resultive_df.loc[resultive_df[feature] < ql, feature] = ql
    resultive_df.drop([feature], axis=1, inplace=True)
    return ql, qr

def add_one_hot(feature, quantile_dict, qd, application_df, resultive_df, 
                application_features_numeric, application_features_categorical):
    cats = np.unique(
            resultive_df[~resultive_df[feature].isna()][feature])
    if quantile_dict is not None:
        cats = quantile_dict[feature]
    qd[feature] = cats
    for cat in cats:
        application_features_numeric.append('application__{}__{}'.format(feature, cat))
        resultive_df['application__{}__{}'.format(feature, cat)] = \
                (resultive_df[feature] == cat).astype(float)
    if application_df[application_df[feature].isna()].shape[0] != 0:
        application_features_numeric.append('application__{}__{}'.format(feature, 'nan'))
        resultive_df['application__{}__{}'.format(feature, 'nan')] = \
                (resultive_df[feature].isna()).astype(float)
    resultive_df.drop([feature], axis=1, inplace=True)
    
def add_NAME_TYPE_SUITE(application_df, resultive_df, 
                        application_features_numeric, application_features_categorical):
    resultive_df.loc[resultive_df['NAME_TYPE_SUITE'] != 'Unaccompanied',
                     'NAME_TYPE_SUITE'] = 'Accompanied'
    resultive_df['application__NAME_TYPE_SUITE__Accompanied'] = \
                (resultive_df['NAME_TYPE_SUITE'] == 'Accompanied').astype(float)
    resultive_df['application__NAME_TYPE_SUITE__Unaccompanied'] = \
                (resultive_df['NAME_TYPE_SUITE'] == 'Unaccompanied').astype(float)
    application_features_numeric.append('application__NAME_TYPE_SUITE__Accompanied')
    application_features_numeric.append('application__NAME_TYPE_SUITE__Unaccompanied')
    resultive_df.drop(['NAME_TYPE_SUITE'], axis=1, inplace=True)
    
def add_OWN_CAR_AGE(application_df, resultive_df, 
                    application_features_numeric, application_features_categorical):
    application_features_numeric.append('application__OWN_CAR_AGE__0_8')
    resultive_df['application__OWN_CAR_AGE__0_8'] = (
            (application_df['OWN_CAR_AGE'] <= 8) &\
            (~application_df['OWN_CAR_AGE'].isna())
    ).astype(float)
    application_features_numeric.append('application__OWN_CAR_AGE__8_12')
    resultive_df['application__OWN_CAR_AGE__8_12'] = (
            (application_df['OWN_CAR_AGE'] > 8) & (application_df['OWN_CAR_AGE'] <= 12) &\
            (~application_df['OWN_CAR_AGE'].isna())
    ).astype(float)
    application_features_numeric.append('application__OWN_CAR_AGE__12_60')
    resultive_df['application__OWN_CAR_AGE__12_60'] = (
            (application_df['OWN_CAR_AGE'] > 12) & (application_df['OWN_CAR_AGE'] < 60) &\
            (~application_df['OWN_CAR_AGE'].isna())
    ).astype(float)
    application_features_numeric.append('application__OWN_CAR_AGE__60_inf')
    resultive_df['application__OWN_CAR_AGE__60_inf'] = (
            (application_df['OWN_CAR_AGE'] >= 60)&\
            (~application_df['OWN_CAR_AGE'].isna())
    ).astype(float)
    application_features_numeric.append('application__OWN_CAR_AGE__nan')
    resultive_df['application__OWN_CAR_AGE__nan'] = (
            (application_df['OWN_CAR_AGE'].isna())
    ).astype(float)
    resultive_df.drop(['OWN_CAR_AGE'], axis=1, inplace=True)

def add_CODE_GENDER(application_df, resultive_df, 
                    application_features_numeric, application_features_categorical):
    resultive_df.loc[resultive_df['CODE_GENDER'] == 'XNA', 'CODE_GENDER'] = 'F'
    application_features_numeric.append('application__CODE_GENDER__male')
    resultive_df['application__CODE_GENDER__male'] = (
            resultive_df['CODE_GENDER'] == 'M').astype(float)
    application_features_numeric.append('application__CODE_GENDER__female')
    resultive_df['application__CODE_GENDER__female'] = (
            resultive_df['CODE_GENDER'] == 'F').astype(float)
    resultive_df.drop(['CODE_GENDER'], axis=1, inplace=True)

def add_FLAG_OWN_REALTY(application_df, resultive_df, 
                        application_features_numeric, application_features_categorical):
    application_features_numeric.append('application__FLAG_OWN_REALTY')
    resultive_df['application__FLAG_OWN_REALTY'] = (
            resultive_df['FLAG_OWN_REALTY'] == 'Y').astype(float)
    resultive_df.drop(['FLAG_OWN_REALTY'], axis=1, inplace=True)

def add_CNT_CHILDREN(application_df, resultive_df, 
                     application_features_numeric, application_features_categorical):
    application_features_numeric.append('application__CNT_CHILDREN__0')
    resultive_df['application__CNT_CHILDREN__0'] = (
            application_df['CNT_CHILDREN'] == 0).astype(float)
    application_features_numeric.append('application__CNT_CHILDREN__1')
    resultive_df['application__CNT_CHILDREN__1'] = (
            application_df['CNT_CHILDREN'] == 1).astype(float)
    application_features_numeric.append('application__CNT_CHILDREN__2_inf')
    resultive_df['application__CNT_CHILDREN__2_inf'] = (
            application_df['CNT_CHILDREN'] >= 2).astype(float)
    resultive_df.drop(['CNT_CHILDREN'], axis=1, inplace=True)    

def add_CNT_FAM_MEMBERS(application_df, resultive_df, 
                        application_features_numeric, application_features_categorical):
    resultive_df['CNT_FAM_MEMBERS'].fillna(1, inplace=True)
    application_features_numeric.append('application__CNT_FAM_MEMBERS__1')
    resultive_df['application__CNT_FAM_MEMBERS__1'] = (
            application_df['CNT_FAM_MEMBERS'] == 1).astype(float)
    application_features_numeric.append('application__CNT_FAM_MEMBERS__2')
    resultive_df['application__CNT_FAM_MEMBERS__2'] = (
            application_df['CNT_FAM_MEMBERS'] == 2).astype(float)
    application_features_numeric.append('application__CNT_FAM_MEMBERS__3')
    resultive_df['application__CNT_FAM_MEMBERS__3'] = (
            application_df['CNT_FAM_MEMBERS'] == 3).astype(float)
    application_features_numeric.append('application__CNT_FAM_MEMBERS__4_inf')
    resultive_df['application__CNT_FAM_MEMBERS__4_inf'] = (
            application_df['CNT_FAM_MEMBERS'] >= 4).astype(float)
    resultive_df.drop(['CNT_FAM_MEMBERS'], axis=1, inplace=True)
    
def add_NAME_INCOME_TYPE(application_df, quantile_dict, qd, resultive_df, 
                         application_features_numeric, application_features_categorical):
    resultive_df.loc[(resultive_df['NAME_INCOME_TYPE'] != 'Commercial associate') &\
                     (resultive_df['NAME_INCOME_TYPE'] != 'Pensioner') &\
                     (resultive_df['NAME_INCOME_TYPE'] != 'State servant') &\
                     (resultive_df['NAME_INCOME_TYPE'] != 'Working'),
                     'NAME_INCOME_TYPE'] = 'Working'
    add_one_hot('NAME_INCOME_TYPE', quantile_dict, qd, application_df, resultive_df, 
                application_features_numeric, application_features_categorical)

def add_ORGANIZATION_TYPE(application_df, resultive_df, 
                         application_features_numeric, application_features_categorical):
    low_prob_orgs = [
        'Bank', 'Culture', 'Industry: type 12', 'Insurance',
        'Military', 'Police', 'Religion', 'School', 'Security Ministries',
        'Trade: type 4', 'Trade: type 6', 'Transport: type 1',
        'University', 'XNA'
    ]
    high_prob_orgs = [
        'Agriculture', 'Business Entity Type 3', 'Cleaning', 'Construction',
        'Industry: type 1', 'Industry: type 13', 'Industry: type 3', 'Industry: type 4',
        'Industry: type 8', 'Mobile', 'Realtor', 'Restaurant', 'Security',
        'Self-employed', 'Trade: type 3', 'Trade: type 7', 'Transport: type 3',
        'Transport: type 4'
    ]
    application_features_numeric.append('application__ORGANIZATION_TYPE__high_default_rate')
    resultive_df['application__ORGANIZATION_TYPE__high_default_rate'] = (
        application_df['ORGANIZATION_TYPE'].isin(high_prob_orgs)).astype(float)
    application_features_numeric.append('application__ORGANIZATION_TYPE__low_default_rate')
    resultive_df['application__ORGANIZATION_TYPE__low_default_rate'] = (
        application_df['ORGANIZATION_TYPE'].isin(low_prob_orgs)).astype(float)
    resultive_df.drop(['ORGANIZATION_TYPE'], axis=1, inplace=True)

def add_AMT_INCOME_TOTAL(application_df, resultive_df, 
                         application_features_numeric, application_features_categorical):
    resultive_df.loc[resultive_df['AMT_INCOME_TOTAL'] > 500000, 'AMT_INCOME_TOTAL'] = 500000
    add_renamed('AMT_INCOME_TOTAL', application_df, resultive_df, 
                application_features_numeric, application_features_categorical)

def add_AMT_CREDIT(application_df, resultive_df, 
                         application_features_numeric, application_features_categorical):
    resultive_df.loc[resultive_df['AMT_CREDIT'] > 2000000, 'AMT_CREDIT'] = 2000000
    add_renamed('AMT_CREDIT', application_df, resultive_df, 
                application_features_numeric, application_features_categorical)

def add_AMT_ANNUITY(application_df, resultive_df, 
                    application_features_numeric, application_features_categorical):
    resultive_df['AMT_ANNUITY'].fillna(resultive_df['AMT_ANNUITY'].mean(), inplace=True)
    resultive_df.loc[resultive_df['AMT_ANNUITY'] > 80000, 'AMT_ANNUITY'] = 80000
    add_renamed('AMT_ANNUITY', application_df, resultive_df, 
                application_features_numeric, application_features_categorical)

def add_AMT_GOODS_PRICE(application_df, resultive_df, 
                        application_features_numeric, application_features_categorical):
    resultive_df.loc[resultive_df['AMT_GOODS_PRICE'] > 200000, 'AMT_GOODS_PRICE'] = 200000
    resultive_df['AMT_GOODS_PRICE'].fillna(resultive_df['AMT_GOODS_PRICE'].mean(), inplace=True)
    add_renamed('AMT_GOODS_PRICE', application_df, resultive_df, 
                application_features_numeric, application_features_categorical)

def add_AMT_REQ_CREDIT_BUREAU_YEAR(application_df, resultive_df, 
                     application_features_numeric, application_features_categorical):
    application_features_numeric.append('application__AMT_REQ_CREDIT_BUREAU_YEAR__6_inf')
    resultive_df['application__AMT_REQ_CREDIT_BUREAU_YEAR__6_inf'] = (
            application_df['AMT_REQ_CREDIT_BUREAU_YEAR'] >= 6).astype(float)
    resultive_df.drop(['AMT_REQ_CREDIT_BUREAU_YEAR'], axis=1, inplace=True)  

def add_NAME_CONTRACT_TYPE(application_df, resultive_df, 
                           application_features_numeric, application_features_categorical):
    application_features_numeric.append('application__NAME_CONTRACT_TYPE__cash')
    resultive_df['application__NAME_CONTRACT_TYPE__cash'] = (
            resultive_df['NAME_CONTRACT_TYPE'] == 'Cash loans').astype(float)
    resultive_df.drop(['NAME_CONTRACT_TYPE'], axis=1, inplace=True)
   
def remove_low_iv_cols(resultive_df):
    low_iv_cols = [
        'application__NAME_FAMILY_STATUS__Unknown',
        'application__NAME_TYPE_SUITE__Accompanied', 
        'application__NAME_TYPE_SUITE__Unaccompanied', 
        'application__OCCUPATION_TYPE__Cleaning staff', 
        'application__OCCUPATION_TYPE__HR staff', 
        'application__OCCUPATION_TYPE__IT staff', 
        'application__OCCUPATION_TYPE__Private service staff', 
        'application__OCCUPATION_TYPE__Realty agents',
        'application__OCCUPATION_TYPE__Secretaries', 
        'application__OCCUPATION_TYPE__Waiters/barmen staff', 
        'application__OWN_CAR_AGE__8_12',
        'application__OWN_CAR_AGE__60_inf', 
        'application__FLAG_OWN_REALTY',
        'application__NAME_FAMILY_STATUS__Separated', 
        'application__FLAG_EMAIL', 
        'application__CNT_FAM_MEMBERS__1',
        'application__NAME_EDUCATION_TYPE__Academic degree',
        'application__NAME_EDUCATION_TYPE__Incomplete higher', 
        'application__NAME_HOUSING_TYPE__Co-op apartment',
        'application__NAME_HOUSING_TYPE__Municipal apartment', 
        'application__NAME_HOUSING_TYPE__Office apartment', 
        'application__OBS_30_CNT_SOCIAL_CIRCLE', 
        'application__DEF_30_CNT_SOCIAL_CIRCLE',
        'application__OBS_60_CNT_SOCIAL_CIRCLE', 
        'application__DEF_60_CNT_SOCIAL_CIRCLE']
    resultive_df.drop(low_iv_cols, axis=1, inplace=True)
    
def remove_high_corrs(resultive_df):
    high_corrs_cols = [
        'application__ELEVATORS_MEDI',
         'application__NONLIVINGAREA_AVG',
         'application__LANDAREA_AVG',
         'application__COMMONAREA_MODE',
         'application__ENTRANCES_MODE',
         'application__ELEVATORS_MODE',
         'application__ENTRANCES_AVG',
         'application__LANDAREA_MODE',
         'application__COMMONAREA_AVG',
         'application__NONLIVINGAPARTMENTS_MEDI',
         'application__BASEMENTAREA_MODE',
         'application__LIVINGAPARTMENTS_AVG',
         'application__YEARS_BUILD_MODE',
         'application__LIVINGAREA_AVG',
         'application__LIVINGAREA_MODE',
         'application__APARTMENTS_MEDI',
         'application__FLOORSMIN_MEDI',
         'application__LIVINGAPARTMENTS_MODE',
         'application__APARTMENTS_MODE',
         'application__YEARS_BEGINEXPLUATATION_MODE',
         'application__NONLIVINGAPARTMENTS_MODE',
         'application__NONLIVINGAREA_MODE',
         'application__BASEMENTAREA_AVG',
         'application__FLOORSMIN_MODE',
         'application__YEARS_BEGINEXPLUATATION_AVG',
         'application__LIVINGAREA_MEDI',
         'application__APARTMENTS_AVG',
         'application__YEARS_BUILD_AVG',
         'application__FLOORSMAX_MODE',
         'application__ELEVATORS_AVG',
         'application__LIVINGAPARTMENTS_MEDI',
         'application__FLOORSMAX_AVG',
         'application__FLOORSMIN_AVG']
    resultive_df.drop(high_corrs_cols, axis=1, inplace=True)

def add_complex_application_features(df):
    new_feature_names = []
    for agg, func in zip(
            ['min', 'max', 'mean', 'nanmedian'], [np.min, np.max, np.mean, np.nanmedian]):
        feature = 'application__EXT_SOURCES_{}'.format(agg)
        new_feature_names.append(feature)
        df[feature] = func(
                df[[
                    'application__EXT_SOURCE_1', 
                    'application__EXT_SOURCE_2', 
                    'application__EXT_SOURCE_3']], axis=1)
    for features in [('AMT_CREDIT', 'AMT_ANNUITY'), ('AMT_CREDIT', 'AMT_GOODS_PRICE'),
                     ('AMT_INCOME_TOTAL', 'AMT_ANNUITY'), ('AMT_INCOME_TOTAL', 'AMT_CREDIT'),
                     ('AMT_INCOME_TOTAL', 'AMT_GOODS_PRICE'), ('AMT_INCOME_TOTAL', 'DAYS_BIRTH'),
                     ('AMT_CREDIT', 'DAYS_BIRTH'), ('AMT_ANNUITY', 'AMT_GOODS_PRICE')]:
        f1, f2 = features
        new_feature_names.append('application__{}_TO_{}'.format(f1, f2))
        df['application__{}_TO_{}'.format(f1, f2)] = df['application__{}'.format(f1)] /\
                                                        df['application__{}'.format(f2)]
    return new_feature_names
    
def build_training_data_for_application(application_df, quantile_dict=None):
    # quantile_dict is None -> train, quantile_dict is not None -> test
    qd = {}
    resultive_df = application_df.copy(deep=True)
    application_features_categorical = []
    application_features_numeric = []
    #1#########################################################################
    # NAME_TYPE_SUITE - manual OHE Encoding
    add_NAME_TYPE_SUITE(application_df, resultive_df, 
                        application_features_numeric, application_features_categorical)
    #2#########################################################################
    # OCCUPATION_TYPE - manual OHE Encoding
    add_one_hot('OCCUPATION_TYPE', quantile_dict, qd, application_df, resultive_df, 
                application_features_numeric, application_features_categorical)
    #3#########################################################################
    # OWN_CAR_AGE - manual OHE Encoding
    add_OWN_CAR_AGE(application_df, resultive_df, 
                    application_features_numeric, application_features_categorical)
    #4#########################################################################
    # CODE_GENDER - manual OHE Encoding
    add_CODE_GENDER(application_df, resultive_df, 
                    application_features_numeric, application_features_categorical)
    #5#########################################################################
    # FLAG_OWN_CAR - drop (add_OWN_CAR_AGE - is enough)
    resultive_df.drop(['FLAG_OWN_CAR'], axis=1, inplace=True)
    #6#########################################################################
    # FLAG_OWN_REALTY - manual OHE Encoding
    add_FLAG_OWN_REALTY(application_df, resultive_df, 
                        application_features_numeric, application_features_categorical)
    #7#########################################################################
    # NAME_FAMILY_STATUS
    add_one_hot('NAME_FAMILY_STATUS', quantile_dict, qd, application_df, resultive_df, 
                application_features_numeric, application_features_categorical)
    #8#########################################################################
    # FLAG_MOBIL - useless
    resultive_df.drop(['FLAG_MOBIL'], axis=1, inplace=True)
    #9#########################################################################
    # FLAG_EMP_PHONE
    add_renamed('FLAG_EMP_PHONE', application_df, resultive_df, 
                application_features_numeric, application_features_categorical)
    #10########################################################################
    # FLAG_WORK_PHONE
    add_renamed('FLAG_WORK_PHONE', application_df, resultive_df, 
                application_features_numeric, application_features_categorical)
    #11########################################################################
    # FLAG_CONT_MOBILE - useless
    resultive_df.drop(['FLAG_CONT_MOBILE'], axis=1, inplace=True)
    #12########################################################################
    # FLAG_PHONE
    add_renamed('FLAG_PHONE', application_df, resultive_df, 
                application_features_numeric, application_features_categorical)
    #13########################################################################
    # FLAG_EMAIL
    add_renamed('FLAG_EMAIL', application_df, resultive_df, 
                application_features_numeric, application_features_categorical)
    #14########################################################################
    # CNT_CHILDREN
    add_CNT_CHILDREN(application_df, resultive_df, 
                     application_features_numeric, application_features_categorical)
    #15########################################################################
    # CNT_FAM_MEMBERS
    add_CNT_FAM_MEMBERS(application_df, resultive_df, 
                        application_features_numeric, application_features_categorical)
    #16########################################################################
    # NAME_INCOME_TYPE
    add_NAME_INCOME_TYPE(application_df, quantile_dict, qd, resultive_df, 
                         application_features_numeric, application_features_categorical)
    #17########################################################################
    # NAME_EDUCATION_TYPE
    add_one_hot('NAME_EDUCATION_TYPE', quantile_dict, qd, application_df, resultive_df, 
                application_features_numeric, application_features_categorical)
    #18########################################################################
    # NAME_HOUSING_TYPE
    add_one_hot('NAME_HOUSING_TYPE', quantile_dict, qd, application_df, resultive_df, 
                application_features_numeric, application_features_categorical)
    #19########################################################################
    # ORGANIZATION_TYPE
    add_ORGANIZATION_TYPE(application_df, resultive_df, 
                         application_features_numeric, application_features_categorical)
    #20########################################################################
    # DAYS_LAST_PHONE_CHANGE - useless
    resultive_df.drop(['DAYS_LAST_PHONE_CHANGE'], axis=1, inplace=True)
    #21########################################################################
    # DAYS_BIRTH
    add_renamed('DAYS_BIRTH', application_df, resultive_df, 
                application_features_numeric, application_features_categorical)
    #22########################################################################
    # DAYS_EMPLOYED -
    resultive_df.drop(['DAYS_EMPLOYED'], axis=1, inplace=True)
    #23########################################################################
    # DAYS_REGISTRATION -
    resultive_df.drop(['DAYS_REGISTRATION'], axis=1, inplace=True)
    #24########################################################################
    # DAYS_ID_PUBLISH -
    resultive_df.drop(['DAYS_ID_PUBLISH'], axis=1, inplace=True)
    #25########################################################################
    # AMT_INCOME_TOTAL
    add_AMT_INCOME_TOTAL(application_df, resultive_df, 
                         application_features_numeric, application_features_categorical)
    #26########################################################################
    # AMT_CREDIT
    add_AMT_CREDIT(application_df, resultive_df, 
                   application_features_numeric, application_features_categorical)
    #27########################################################################
    # AMT_ANNUITY
    add_AMT_ANNUITY(application_df, resultive_df, 
                    application_features_numeric, application_features_categorical)
    #28########################################################################
    # AMT_GOODS_PRICE
    add_AMT_GOODS_PRICE(application_df, resultive_df, 
                        application_features_numeric, application_features_categorical)
    #29########################################################################
    # AMT_REQ_CREDIT_BUREAU_HOUR -
    resultive_df.drop(['AMT_REQ_CREDIT_BUREAU_HOUR'], axis=1, inplace=True)
    #30########################################################################
    # AMT_REQ_CREDIT_BUREAU_DAY -
    resultive_df.drop(['AMT_REQ_CREDIT_BUREAU_DAY'], axis=1, inplace=True)
    #31########################################################################
    # AMT_REQ_CREDIT_BUREAU_WEEK -
    resultive_df.drop(['AMT_REQ_CREDIT_BUREAU_WEEK'], axis=1, inplace=True)
    #32########################################################################
    # AMT_REQ_CREDIT_BUREAU_MON -
    resultive_df.drop(['AMT_REQ_CREDIT_BUREAU_MON'], axis=1, inplace=True)
    #33########################################################################
    # AMT_REQ_CREDIT_BUREAU_QRT -
    resultive_df.drop(['AMT_REQ_CREDIT_BUREAU_QRT'], axis=1, inplace=True)
    #34########################################################################
    # AMT_REQ_CREDIT_BUREAU_YEAR
    add_AMT_REQ_CREDIT_BUREAU_YEAR(application_df, resultive_df, 
                     application_features_numeric, application_features_categorical)
    #35########################################################################
    # APARTMENTS_AVG
    ql, qr = add_quantile_nan('APARTMENTS_AVG', quantile_dict, application_df, resultive_df, 
                              application_features_numeric, application_features_categorical)
    qd['APARTMENTS_AVG'] = (ql, qr)
    #36########################################################################
    # BASEMENTAREA_AVG
    ql, qr = add_quantile_nan('BASEMENTAREA_AVG', quantile_dict, application_df, resultive_df, 
                              application_features_numeric, application_features_categorical)
    qd['BASEMENTAREA_AVG'] = (ql, qr)
    #37########################################################################
    # YEARS_BEGINEXPLUATATION_AVG
    ql, qr = add_quantile_nan('YEARS_BEGINEXPLUATATION_AVG', quantile_dict, application_df, resultive_df, 
                              application_features_numeric, application_features_categorical)
    qd['YEARS_BEGINEXPLUATATION_AVG'] = (ql, qr)
    #38########################################################################
    # YEARS_BUILD_AVG
    ql, qr = add_quantile_nan('YEARS_BUILD_AVG', quantile_dict, application_df, resultive_df, 
                              application_features_numeric, application_features_categorical)
    qd['YEARS_BUILD_AVG'] = (ql, qr)
    #39########################################################################
    # COMMONAREA_AVG
    ql, qr = add_quantile_nan('COMMONAREA_AVG', quantile_dict, application_df, resultive_df, 
                              application_features_numeric, application_features_categorical)
    qd['COMMONAREA_AVG'] = (ql, qr)
    #40########################################################################
    # ELEVATORS_AVG
    ql, qr = add_quantile_nan('ELEVATORS_AVG', quantile_dict, application_df, resultive_df, 
                              application_features_numeric, application_features_categorical)
    qd['ELEVATORS_AVG'] = (ql, qr)
    #41########################################################################
    # FLOORSMAX_AVG
    ql, qr = add_quantile_nan('FLOORSMAX_AVG', quantile_dict, application_df, resultive_df, 
                              application_features_numeric, application_features_categorical)
    qd['FLOORSMAX_AVG'] = (ql, qr)
    #42########################################################################
    # FLOORSMIN_AVG
    ql, qr = add_quantile_nan('FLOORSMIN_AVG', quantile_dict, application_df, resultive_df, 
                              application_features_numeric, application_features_categorical)
    qd['FLOORSMIN_AVG'] = (ql, qr)
    #43########################################################################
    # LANDAREA_AVG
    ql, qr = add_quantile_nan('LANDAREA_AVG', quantile_dict, application_df, resultive_df, 
                              application_features_numeric, application_features_categorical)
    qd['LANDAREA_AVG'] = (ql, qr)
    #44########################################################################
    # LIVINGAPARTMENTS_AVG
    ql, qr = add_quantile_nan('LIVINGAPARTMENTS_AVG', quantile_dict, application_df, resultive_df, 
                              application_features_numeric, application_features_categorical)
    qd['LIVINGAPARTMENTS_AVG'] = (ql, qr)
    #45########################################################################
    # LIVINGAREA_AVG
    ql, qr = add_quantile_nan('LIVINGAREA_AVG', quantile_dict, application_df, resultive_df, 
                              application_features_numeric, application_features_categorical)
    qd['LIVINGAREA_AVG'] = (ql, qr)
    #46########################################################################
    # NONLIVINGAPARTMENTS_AVG
    ql, qr = add_quantile_nan('NONLIVINGAPARTMENTS_AVG', quantile_dict, application_df, resultive_df, 
                              application_features_numeric, application_features_categorical)
    qd['NONLIVINGAPARTMENTS_AVG'] = (ql, qr)
    #47########################################################################
    # NONLIVINGAREA_AVG
    ql, qr = add_quantile_nan('NONLIVINGAREA_AVG', quantile_dict, application_df, resultive_df, 
                              application_features_numeric, application_features_categorical)
    qd['NONLIVINGAREA_AVG'] = (ql, qr)
    #48########################################################################
    # APARTMENTS_MODE
    ql, qr = add_quantile_nan('APARTMENTS_MODE', quantile_dict, application_df, resultive_df, 
                              application_features_numeric, application_features_categorical)
    qd['APARTMENTS_MODE'] = (ql, qr)
    #49########################################################################
    # BASEMENTAREA_MODE
    ql, qr = add_quantile_nan('BASEMENTAREA_MODE', quantile_dict, application_df, resultive_df, 
                              application_features_numeric, application_features_categorical)
    qd['BASEMENTAREA_MODE'] = (ql, qr)
    #51########################################################################
    # YEARS_BEGINEXPLUATATION_MODE
    ql, qr = add_quantile_nan('YEARS_BEGINEXPLUATATION_MODE', quantile_dict, application_df, resultive_df, 
                              application_features_numeric, application_features_categorical)
    qd['YEARS_BEGINEXPLUATATION_MODE'] = (ql, qr)
    #52########################################################################
    # YEARS_BUILD_MODE
    ql, qr = add_quantile_nan('YEARS_BUILD_MODE', quantile_dict, application_df, resultive_df, 
                              application_features_numeric, application_features_categorical)
    qd['YEARS_BUILD_MODE'] = (ql, qr)
    #53########################################################################
    # COMMONAREA_MODE
    ql, qr = add_quantile_nan('COMMONAREA_MODE', quantile_dict, application_df, resultive_df, 
                              application_features_numeric, application_features_categorical)
    qd['COMMONAREA_MODE'] = (ql, qr)
    #54########################################################################
    # ELEVATORS_MODE
    ql, qr = add_quantile_nan('ELEVATORS_MODE', quantile_dict, application_df, resultive_df, 
                              application_features_numeric, application_features_categorical)
    qd['ELEVATORS_MODE'] = (ql, qr)
    #55########################################################################
    # ENTRANCES_MODE
    ql, qr = add_quantile_nan('ENTRANCES_MODE', quantile_dict, application_df, resultive_df, 
                              application_features_numeric, application_features_categorical)
    qd['ENTRANCES_MODE'] = (ql, qr)
    #56########################################################################
    # FLOORSMAX_MODE
    ql, qr = add_quantile_nan('FLOORSMAX_MODE', quantile_dict, application_df, resultive_df, 
                              application_features_numeric, application_features_categorical)
    qd['FLOORSMAX_MODE'] = (ql, qr)
    #57########################################################################
    # FLOORSMIN_MODE
    ql, qr = add_quantile_nan('FLOORSMIN_MODE', quantile_dict, application_df, resultive_df, 
                              application_features_numeric, application_features_categorical)
    qd['FLOORSMIN_MODE'] = (ql, qr)
    #58########################################################################
    # LANDAREA_MODE
    ql, qr = add_quantile_nan('LANDAREA_MODE', quantile_dict, application_df, resultive_df, 
                              application_features_numeric, application_features_categorical)
    qd['LANDAREA_MODE'] = (ql, qr)
    #59########################################################################
    # LIVINGAPARTMENTS_MODE
    ql, qr = add_quantile_nan('LIVINGAPARTMENTS_MODE', quantile_dict, application_df, resultive_df, 
                              application_features_numeric, application_features_categorical)
    qd['LIVINGAPARTMENTS_MODE'] = (ql, qr)
    #61########################################################################
    # LIVINGAREA_MODE
    ql, qr = add_quantile_nan('LIVINGAREA_MODE', quantile_dict, application_df, resultive_df, 
                              application_features_numeric, application_features_categorical)
    qd['LIVINGAREA_MODE'] = (ql, qr)
    #62########################################################################
    # NONLIVINGAPARTMENTS_MODE
    ql, qr = add_quantile_nan('NONLIVINGAPARTMENTS_MODE', quantile_dict, application_df, resultive_df, 
                              application_features_numeric, application_features_categorical)
    qd['NONLIVINGAPARTMENTS_MODE'] = (ql, qr)
    #63########################################################################
    # NONLIVINGAREA_MODE
    ql, qr = add_quantile_nan('NONLIVINGAREA_MODE', quantile_dict, application_df, resultive_df, 
                              application_features_numeric, application_features_categorical)
    qd['NONLIVINGAREA_MODE'] = (ql, qr)
    #64########################################################################
    # APARTMENTS_MEDI
    ql, qr = add_quantile_nan('APARTMENTS_MEDI', quantile_dict, application_df, resultive_df, 
                              application_features_numeric, application_features_categorical)
    qd['APARTMENTS_MEDI'] = (ql, qr)
    #65########################################################################
    # BASEMENTAREA_MEDI
    ql, qr = add_quantile_nan('BASEMENTAREA_MEDI', quantile_dict, application_df, resultive_df, 
                              application_features_numeric, application_features_categorical)
    qd['BASEMENTAREA_MEDI'] = (ql, qr)
    #66########################################################################
    # YEARS_BEGINEXPLUATATION_MEDI
    ql, qr = add_quantile_nan('YEARS_BEGINEXPLUATATION_MEDI', quantile_dict, application_df, resultive_df, 
                              application_features_numeric, application_features_categorical)
    qd['YEARS_BEGINEXPLUATATION_MEDI'] = (ql, qr)
    #67########################################################################
    # YEARS_BUILD_MEDI
    ql, qr = add_quantile_nan('YEARS_BUILD_MEDI', quantile_dict, application_df, resultive_df, 
                              application_features_numeric, application_features_categorical)
    qd['YEARS_BUILD_MEDI'] = (ql, qr)
    #68########################################################################
    # COMMONAREA_MEDI
    ql, qr = add_quantile_nan('COMMONAREA_MEDI', quantile_dict, application_df, resultive_df, 
                              application_features_numeric, application_features_categorical)
    qd['COMMONAREA_MEDI'] = (ql, qr)
    #69########################################################################
    # ELEVATORS_MEDI
    ql, qr = add_quantile_nan('ELEVATORS_MEDI', quantile_dict, application_df, resultive_df, 
                              application_features_numeric, application_features_categorical)
    qd['ELEVATORS_MEDI'] = (ql, qr)
    #70########################################################################
    # ENTRANCES_MEDI
    ql, qr = add_quantile_nan('ENTRANCES_MEDI', quantile_dict, application_df, resultive_df, 
                              application_features_numeric, application_features_categorical)
    qd['ENTRANCES_MEDI'] = (ql, qr)
    #71########################################################################
    # FLOORSMAX_MEDI
    ql, qr = add_quantile_nan('FLOORSMAX_MEDI', quantile_dict, application_df, resultive_df, 
                              application_features_numeric, application_features_categorical)
    qd['FLOORSMAX_MEDI'] = (ql, qr)
    #72########################################################################
    # FLOORSMIN_MEDI
    ql, qr = add_quantile_nan('FLOORSMIN_MEDI', quantile_dict, application_df, resultive_df, 
                              application_features_numeric, application_features_categorical)
    qd['FLOORSMIN_MEDI'] = (ql, qr)
    #73########################################################################
    # LANDAREA_MEDI
    ql, qr = add_quantile_nan('LANDAREA_MEDI', quantile_dict, application_df, resultive_df, 
                              application_features_numeric, application_features_categorical)
    qd['LANDAREA_MEDI'] = (ql, qr)
    #74########################################################################
    # LIVINGAPARTMENTS_MEDI
    ql, qr = add_quantile_nan('LIVINGAPARTMENTS_MEDI', quantile_dict, application_df, resultive_df, 
                              application_features_numeric, application_features_categorical)
    qd['LIVINGAPARTMENTS_MEDI'] = (ql, qr)
    #75########################################################################
    # LIVINGAREA_MEDI
    ql, qr = add_quantile_nan('LIVINGAREA_MEDI', quantile_dict, application_df, resultive_df, 
                              application_features_numeric, application_features_categorical)
    qd['LIVINGAREA_MEDI'] = (ql, qr)
    #76########################################################################
    # NONLIVINGAPARTMENTS_MEDI
    ql, qr = add_quantile_nan('NONLIVINGAPARTMENTS_MEDI', quantile_dict, application_df, resultive_df, 
                              application_features_numeric, application_features_categorical)
    qd['NONLIVINGAPARTMENTS_MEDI'] = (ql, qr)
    #77########################################################################
    # NONLIVINGAREA_MEDI
    ql, qr = add_quantile_nan('NONLIVINGAREA_MEDI', quantile_dict, application_df, resultive_df, 
                              application_features_numeric, application_features_categorical)
    qd['NONLIVINGAREA_MEDI'] = (ql, qr)
    #78########################################################################
    # FONDKAPREMONT_MODE
    #79########################################################################
    # HOUSETYPE_MODE
    #80########################################################################
    # TOTALAREA_MODE
    ql, qr = add_quantile_nan('TOTALAREA_MODE', quantile_dict, application_df, resultive_df, 
                              application_features_numeric, application_features_categorical)
    qd['TOTALAREA_MODE'] = (ql, qr)
    #81########################################################################
    # WALLSMATERIAL_MODE
    #82########################################################################
    # EMERGENCYSTATE_MODE
    #83########################################################################
    # FLAG_DOCUMENT_3
    add_renamed('FLAG_DOCUMENT_3', application_df, resultive_df,
                application_features_numeric, application_features_categorical)
    #84########################################################################
    # EXT_SOURCE_1
    add_renamed_nan('EXT_SOURCE_1', application_df, resultive_df,
                application_features_numeric, application_features_categorical)
    #85########################################################################
    # EXT_SOURCE_2
    add_renamed_nan('EXT_SOURCE_2', application_df, resultive_df,
                application_features_numeric, application_features_categorical)
    #86########################################################################
    # EXT_SOURCE_3
    add_renamed_nan('EXT_SOURCE_3', application_df, resultive_df,
                application_features_numeric, application_features_categorical)
    #87########################################################################
    # OBS_30_CNT_SOCIAL_CIRCLE
    ql, qr = add_quantile_nan('OBS_30_CNT_SOCIAL_CIRCLE', quantile_dict, application_df,
                              resultive_df, application_features_numeric, 
                              application_features_categorical)
    qd['OBS_30_CNT_SOCIAL_CIRCLE'] = (ql, qr)
    #88########################################################################
    # DEF_30_CNT_SOCIAL_CIRCLE
    ql, qr = add_quantile_nan('DEF_30_CNT_SOCIAL_CIRCLE', quantile_dict, application_df,
                              resultive_df, application_features_numeric, 
                              application_features_categorical)
    qd['DEF_30_CNT_SOCIAL_CIRCLE'] = (ql, qr)
    #89########################################################################
    # OBS_60_CNT_SOCIAL_CIRCLE
    ql, qr = add_quantile_nan('OBS_60_CNT_SOCIAL_CIRCLE', quantile_dict, application_df,
                              resultive_df, application_features_numeric, 
                              application_features_categorical)
    qd['OBS_60_CNT_SOCIAL_CIRCLE'] = (ql, qr)
    #90########################################################################
    # DEF_60_CNT_SOCIAL_CIRCLE
    ql, qr = add_quantile_nan('DEF_60_CNT_SOCIAL_CIRCLE', quantile_dict, application_df,
                              resultive_df, application_features_numeric, 
                              application_features_categorical)
    qd['DEF_60_CNT_SOCIAL_CIRCLE'] = (ql, qr)
    #91########################################################################
    # REGION_RATING_CLIENT
    add_renamed('REGION_RATING_CLIENT', application_df, resultive_df,
                application_features_numeric, application_features_categorical)
    #92########################################################################
    # REGION_RATING_CLIENT_W_CITY
    add_renamed('REGION_RATING_CLIENT_W_CITY', application_df, resultive_df,
                application_features_numeric, application_features_categorical)
    #93########################################################################
    # REG_REGION_NOT_LIVE_REGION -
    #94########################################################################
    # REG_REGION_NOT_WORK_REGION -
    #95########################################################################
    # LIVE_REGION_NOT_WORK_REGION -
    #96########################################################################
    # REG_CITY_NOT_LIVE_CITY
    add_renamed('REG_CITY_NOT_LIVE_CITY', application_df, resultive_df,
                application_features_numeric, application_features_categorical)
    #97########################################################################
    # REG_CITY_NOT_WORK_CITY
    add_renamed('REG_CITY_NOT_WORK_CITY', application_df, resultive_df,
                application_features_numeric, application_features_categorical)
    #98########################################################################
    # LIVE_CITY_NOT_WORK_CITY
    add_renamed('LIVE_CITY_NOT_WORK_CITY', application_df, resultive_df,
                application_features_numeric, application_features_categorical)
    #99########################################################################
    # REGION_POPULATION_RELATIVE
    add_renamed('REGION_POPULATION_RELATIVE', application_df, resultive_df,
                application_features_numeric, application_features_categorical)
    #100#######################################################################
    # ENTRANCES_AVG
    ql, qr = add_quantile_nan('ENTRANCES_AVG', quantile_dict, application_df,
                              resultive_df, application_features_numeric, 
                              application_features_categorical)
    qd['ENTRANCES_AVG'] = (ql, qr)
    #101#######################################################################
    # NAME_CONTRACT_TYPE
    add_NAME_CONTRACT_TYPE(application_df, resultive_df, 
                           application_features_numeric, application_features_categorical)
    #102#######################################################################
    #103#######################################################################
    #104#######################################################################
    #105#######################################################################
    #106#######################################################################
    
    ###########################################################################
    features = add_complex_application_features(resultive_df)
    resultive_df = resultive_df[application_features_categorical +\
                                application_features_numeric + features]
    remove_low_iv_cols(resultive_df)
    remove_high_corrs(resultive_df)
    ###########################################################################
    return (resultive_df, qd)

BUREAU_CATEGORICAL_FEATURES = [
    'CREDIT_ACTIVE',
    'CREDIT_CURRENCY',
    'CREDIT_TYPE',
]
BUREAU_NUMERIC_FEATURES = [
    'SK_ID_CURR',
    'SK_ID_BUREAU',
    'DAYS_CREDIT',
    'CREDIT_DAY_OVERDUE',
    'DAYS_CREDIT_ENDDATE',
    'DAYS_ENDDATE_FACT',
    'AMT_CREDIT_MAX_OVERDUE',
    'CNT_CREDIT_PROLONG',
    'AMT_CREDIT_SUM',
    'AMT_CREDIT_SUM_DEBT',
    'AMT_CREDIT_SUM_LIMIT',
    'AMT_CREDIT_SUM_OVERDUE',
    'DAYS_CREDIT_UPDATE',
    'AMT_ANNUITY',
]

def bureau_remove_low_iv(df):
    low_iv_features = [
        'bureau___CREDIT_DAY_OVERDUE_median',
        'bureau___CREDIT_DAY_OVERDUE_mean',
        'bureau___CREDIT_DAY_OVERDUE_max',
        'bureau___CREDIT_DAY_OVERDUE_min',
        'bureau___CREDIT_DAY_OVERDUE_sum',
        'bureau___AMT_CREDIT_MAX_OVERDUE_median',
        'bureau___AMT_CREDIT_MAX_OVERDUE_min',
        'bureau___CNT_CREDIT_PROLONG_median',
        'bureau___CNT_CREDIT_PROLONG_mean',
        'bureau___CNT_CREDIT_PROLONG_max',
        'bureau___CNT_CREDIT_PROLONG_min',
        'bureau___CNT_CREDIT_PROLONG_sum',
        'bureau___AMT_CREDIT_SUM_DEBT_min',
        'bureau___AMT_CREDIT_SUM_LIMIT_median',
        'bureau___AMT_CREDIT_SUM_LIMIT_mean',
        'bureau___AMT_CREDIT_SUM_LIMIT_max',
        'bureau___AMT_CREDIT_SUM_LIMIT_sum',
        'bureau___AMT_CREDIT_SUM_OVERDUE_median',
        'bureau___AMT_CREDIT_SUM_OVERDUE_mean',
        'bureau___AMT_CREDIT_SUM_OVERDUE_max',
        'bureau___AMT_CREDIT_SUM_OVERDUE_min',
        'bureau___AMT_CREDIT_SUM_OVERDUE_sum',
        'bureau___AMT_ANNUITY_min',
        'bureau___AMT_ANNUITY_sum',
    ]
    df.drop(low_iv_features, axis=1, inplace=True)
    return low_iv_features

def add_bureau_features(df, bureau):
    bureau_features = []
    # bureau___previous_credits_count
    bureau_features.append('bureau___previous_credits_count')
    bureau_previous_credits_count = bureau.groupby('SK_ID_CURR', as_index=False)['SK_ID_BUREAU'].count().rename(
        columns = {'SK_ID_BUREAU': 'bureau___previous_credits_count'})
    df = df.merge(bureau_previous_credits_count, on='SK_ID_CURR', how='left')
    df['bureau___previous_credits_count'] = df['bureau___previous_credits_count'].fillna(0)
    # bureau___previous_credits_active_count
    bureau_features.append('bureau___previous_credits_active_count')
    bureau_previous_credits_active_count = bureau[bureau['CREDIT_ACTIVE'] == 'Active'].groupby('SK_ID_CURR', as_index=False)['SK_ID_BUREAU'].count().rename(
        columns = {'SK_ID_BUREAU': 'bureau___previous_credits_active_count'})
    df = df.merge(bureau_previous_credits_active_count, on='SK_ID_CURR', how='left')
    df['bureau___previous_credits_active_count'] = df['bureau___previous_credits_active_count'].fillna(0)
    # bureau___previous_credits_closed_count
    bureau_features.append('bureau___previous_credits_closed_count')
    bureau_previous_credits_closed_count = bureau[bureau['CREDIT_ACTIVE'] == 'Active'].groupby('SK_ID_CURR', as_index=False)['SK_ID_BUREAU'].count().rename(
        columns = {'SK_ID_BUREAU': 'bureau___previous_credits_closed_count'})
    df = df.merge(bureau_previous_credits_closed_count, on='SK_ID_CURR', how='left')
    df['bureau___previous_credits_closed_count'] = df['bureau___previous_credits_closed_count'].fillna(0)
    # bureau___previous_credits_active_rate
    bureau_features.append('bureau___previous_credits_active_rate')
    df['bureau___previous_credits_active_rate'] = df['bureau___previous_credits_active_count'] / df['bureau___previous_credits_count']
    # numerical statistics
    new_cols = []
    bureau_numerical_stats = bureau[BUREAU_NUMERIC_FEATURES].groupby('SK_ID_CURR', as_index = False).agg(['median', 'mean', 'max', 'min', 'sum']).reset_index()
    for feature in bureau_numerical_stats.columns.levels[0]:
        if feature == 'SK_ID_CURR':
            continue
        for stat in bureau_numerical_stats.columns.levels[1][:-1]:
            new_cols.append('bureau___{}_{}'.format(feature, stat))
    bureau_numerical_stats.columns = ['SK_ID_CURR'] + new_cols
    bureau_features = bureau_features + new_cols
    df = df.merge(bureau_numerical_stats, on='SK_ID_CURR', how='left')
    # bureau___end_date_passed_credits_rate
    bureau_features.append('bureau___end_date_passed_credits_rate')
    end_date_passed_credits_count = bureau[bureau['DAYS_CREDIT_ENDDATE'] < 0].groupby('SK_ID_CURR', as_index=False)['SK_ID_BUREAU'].count().rename(
        columns = {'SK_ID_BUREAU': 'bureau___end_date_passed_credits_count'})
    df = df.merge(end_date_passed_credits_count, on='SK_ID_CURR', how='left')
    df['bureau___end_date_passed_credits_count'] = df['bureau___end_date_passed_credits_count'].fillna(0)
    df['bureau___end_date_passed_credits_rate'] = df['bureau___end_date_passed_credits_count'] / df['bureau___previous_credits_count']
    deleted_cols = bureau_remove_low_iv(df)
    bureau_features = list(set(bureau_features) - set(deleted_cols))
    return df, bureau_features
    
def ccb_remove_low_iv(df):
    low_iv_features = [
        'ccb__AMT_CREDIT_LIMIT_ACTUAL_median',
        'ccb__AMT_CREDIT_LIMIT_ACTUAL_min',
        'ccb__AMT_CREDIT_LIMIT_ACTUAL_sum',
        'ccb__AMT_DRAWINGS_ATM_CURRENT_median',
        'ccb__AMT_DRAWINGS_ATM_CURRENT_mean',
        'ccb__AMT_DRAWINGS_ATM_CURRENT_max',
        'ccb__AMT_DRAWINGS_ATM_CURRENT_sum',
        'ccb__AMT_DRAWINGS_CURRENT_median',
        'ccb__AMT_DRAWINGS_CURRENT_mean',
        'ccb__AMT_DRAWINGS_CURRENT_max',
        'ccb__AMT_DRAWINGS_CURRENT_min',
        'ccb__AMT_DRAWINGS_CURRENT_sum',
        'ccb__AMT_DRAWINGS_OTHER_CURRENT_median',
        'ccb__AMT_DRAWINGS_OTHER_CURRENT_mean',
        'ccb__AMT_DRAWINGS_OTHER_CURRENT_max',
        'ccb__AMT_DRAWINGS_OTHER_CURRENT_min',
        'ccb__AMT_DRAWINGS_OTHER_CURRENT_sum',
        'ccb__AMT_DRAWINGS_POS_CURRENT_median',
        'ccb__AMT_DRAWINGS_POS_CURRENT_mean',
        'ccb__AMT_DRAWINGS_POS_CURRENT_max',
        'ccb__AMT_DRAWINGS_POS_CURRENT_min',
        'ccb__AMT_DRAWINGS_POS_CURRENT_sum',
        'ccb__AMT_INST_MIN_REGULARITY_median',
        'ccb__AMT_INST_MIN_REGULARITY_mean',
        'ccb__AMT_INST_MIN_REGULARITY_max',
        'ccb__AMT_INST_MIN_REGULARITY_min',
        'ccb__AMT_INST_MIN_REGULARITY_sum',
        'ccb__AMT_PAYMENT_CURRENT_median',
        'ccb__AMT_PAYMENT_CURRENT_mean',
        'ccb__AMT_PAYMENT_CURRENT_max',
        'ccb__AMT_PAYMENT_CURRENT_min',
        'ccb__AMT_PAYMENT_CURRENT_sum',
        'ccb__AMT_PAYMENT_TOTAL_CURRENT_median',
        'ccb__AMT_PAYMENT_TOTAL_CURRENT_mean',
        'ccb__AMT_PAYMENT_TOTAL_CURRENT_max',
        'ccb__AMT_PAYMENT_TOTAL_CURRENT_min',
        'ccb__AMT_PAYMENT_TOTAL_CURRENT_sum',
        'ccb__AMT_RECEIVABLE_PRINCIPAL_median',
        'ccb__AMT_RECEIVABLE_PRINCIPAL_mean',
        'ccb__AMT_RECEIVABLE_PRINCIPAL_max',
        'ccb__AMT_RECEIVABLE_PRINCIPAL_min',
        'ccb__AMT_RECEIVABLE_PRINCIPAL_sum',
        'ccb__AMT_RECIVABLE_median',
        'ccb__AMT_RECIVABLE_mean',
        'ccb__AMT_RECIVABLE_max',
        'ccb__AMT_RECIVABLE_sum',
        'ccb__AMT_TOTAL_RECEIVABLE_median',
        'ccb__AMT_TOTAL_RECEIVABLE_mean',
        'ccb__AMT_TOTAL_RECEIVABLE_max',
        'ccb__AMT_TOTAL_RECEIVABLE_sum',
        'ccb__CNT_DRAWINGS_ATM_CURRENT_mean',
        'ccb__CNT_DRAWINGS_ATM_CURRENT_sum',
        'ccb__CNT_DRAWINGS_CURRENT_median',
        'ccb__CNT_DRAWINGS_CURRENT_mean',
        'ccb__CNT_DRAWINGS_CURRENT_max',
        'ccb__CNT_DRAWINGS_CURRENT_sum',
        'ccb__CNT_DRAWINGS_OTHER_CURRENT_median',
        'ccb__CNT_DRAWINGS_OTHER_CURRENT_mean',
        'ccb__CNT_DRAWINGS_OTHER_CURRENT_max',
        'ccb__CNT_DRAWINGS_OTHER_CURRENT_min',
        'ccb__CNT_DRAWINGS_OTHER_CURRENT_sum',
        'ccb__CNT_DRAWINGS_POS_CURRENT_median',
        'ccb__CNT_DRAWINGS_POS_CURRENT_mean',
        'ccb__CNT_DRAWINGS_POS_CURRENT_max',
        'ccb__CNT_DRAWINGS_POS_CURRENT_sum',
        'ccb__CNT_INSTALMENT_MATURE_CUM_median',
        'ccb__CNT_INSTALMENT_MATURE_CUM_mean',
        'ccb__CNT_INSTALMENT_MATURE_CUM_max',
        'ccb__CNT_INSTALMENT_MATURE_CUM_sum',
        'ccb__SK_DPD_median',
        'ccb__SK_DPD_mean',
        'ccb__SK_DPD_max',
        'ccb__SK_DPD_min',
        'ccb__SK_DPD_sum',
        'ccb__SK_DPD_DEF_median',
        'ccb__SK_DPD_DEF_mean',
        'ccb__SK_DPD_DEF_min',
        'ccb__SK_DPD_DEF_sum',
    ]
    df.drop(low_iv_features, axis=1, inplace=True)
    return low_iv_features
    
def add_ccb_features(df, ccb):
    ccb_cols = []
    ccb_cols.append('ccb__previous_credits_count')
    ccb1 = ccb.groupby('SK_ID_CURR', as_index=False)['SK_ID_PREV'].count().rename(
        columns = {'SK_ID_PREV': 'ccb__previous_credits_count'})
    df = df.merge(ccb1, on='SK_ID_CURR', how='left')
    df['ccb__previous_credits_count'] = df['ccb__previous_credits_count'].fillna(0)
    CCB_NUMERIC_FEATURES = [
        'SK_ID_CURR',
        'AMT_CREDIT_LIMIT_ACTUAL',
        'AMT_DRAWINGS_ATM_CURRENT',
        'AMT_DRAWINGS_CURRENT',
        'AMT_DRAWINGS_OTHER_CURRENT',
        'AMT_DRAWINGS_POS_CURRENT',
        'AMT_INST_MIN_REGULARITY',
        'AMT_PAYMENT_CURRENT',
        'AMT_PAYMENT_TOTAL_CURRENT',
        'AMT_RECEIVABLE_PRINCIPAL',
        'AMT_RECIVABLE',
        'AMT_TOTAL_RECEIVABLE',
        'CNT_DRAWINGS_ATM_CURRENT',
        'CNT_DRAWINGS_CURRENT',
        'CNT_DRAWINGS_OTHER_CURRENT',
        'CNT_DRAWINGS_POS_CURRENT',
        'CNT_INSTALMENT_MATURE_CUM',
        'SK_DPD',
        'SK_DPD_DEF',
    ]
    ccb_numerical_stats = ccb[CCB_NUMERIC_FEATURES].\
        groupby('SK_ID_CURR', as_index = False).\
        agg(['median', 'mean', 'max', 'min', 'sum']).reset_index()
    new_cols = []
    for feature in ccb_numerical_stats.columns.levels[0]:
        if feature == 'SK_ID_CURR':
            continue
        for stat in ccb_numerical_stats.columns.levels[1][:-1]:
            new_cols.append('ccb__{}_{}'.format(feature, stat))
    
    for feature in ccb_numerical_stats.columns:  
        ql, qr = ccb_numerical_stats[feature].quantile(0.01), ccb_numerical_stats[feature].quantile(0.99)
        ccb_numerical_stats.loc[ccb_numerical_stats[feature] > qr, feature] = qr
        ccb_numerical_stats.loc[ccb_numerical_stats[feature] < ql, feature] = ql
    ccb_numerical_stats.columns = ['SK_ID_CURR'] + new_cols
    df = df.merge(ccb_numerical_stats, on='SK_ID_CURR', how='left')
    df[new_cols] = df[new_cols].fillna(0)
    ccb_cols += new_cols
    deleted_cols = ccb_remove_low_iv(df)
    ccb_cols = list(set(ccb_cols) - set(deleted_cols))
    return df, ccb_cols

def pos_cash_remove_low_iv(df):
    low_iv_features = [
        'pos_cash__CNT_INSTALMENT_max',
        'pos_cash__CNT_INSTALMENT_FUTURE_max',
        'pos_cash__SK_DPD_median',
        'pos_cash__SK_DPD_mean',
        'pos_cash__SK_DPD_max',
        'pos_cash__SK_DPD_min',
        'pos_cash__SK_DPD_sum',
        'pos_cash__SK_DPD_DEF_median',
        'pos_cash__SK_DPD_DEF_mean',
        'pos_cash__SK_DPD_DEF_min',
        'pos_cash__SK_DPD_DEF_sum',
    ]
    df.drop(low_iv_features, axis=1, inplace=True)
    return low_iv_features

def add_pos_cash_features(df, pos_cash):
    pos_cash_cols = []
    pos_cash_cols.append('pos_cash__previous_credits_count')
    pos_cash1 = pos_cash.groupby('SK_ID_CURR', as_index=False)['SK_ID_PREV'].count().rename(
        columns = {'SK_ID_PREV': 'pos_cash__previous_credits_count'})
    df = df.merge(pos_cash1, on='SK_ID_CURR', how='left')
    df['pos_cash__previous_credits_count'] = df['pos_cash__previous_credits_count'].fillna(0)
    pos_cash_NUMERIC_FEATURES = [
        'SK_ID_CURR',
        'MONTHS_BALANCE',
        'CNT_INSTALMENT',
        'CNT_INSTALMENT_FUTURE',
        'SK_DPD',
        'SK_DPD_DEF',
    ]
    pos_cash_numerical_stats = pos_cash[pos_cash_NUMERIC_FEATURES].\
        groupby('SK_ID_CURR', as_index = False).\
        agg(['median', 'mean', 'max', 'min', 'sum']).reset_index()
    new_cols = []
    for feature in pos_cash_numerical_stats.columns.levels[0]:
        if feature == 'SK_ID_CURR':
            continue
        for stat in pos_cash_numerical_stats.columns.levels[1][:-1]:
            new_cols.append('pos_cash__{}_{}'.format(feature, stat))
    
    for feature in pos_cash_numerical_stats.columns:  
        ql, qr = pos_cash_numerical_stats[feature].quantile(0.01), pos_cash_numerical_stats[feature].quantile(0.99)
        pos_cash_numerical_stats.loc[pos_cash_numerical_stats[feature] > qr, feature] = qr
        pos_cash_numerical_stats.loc[pos_cash_numerical_stats[feature] < ql, feature] = ql
    pos_cash_numerical_stats.columns = ['SK_ID_CURR'] + new_cols
    df = df.merge(pos_cash_numerical_stats, on='SK_ID_CURR', how='left')
    df[new_cols] = df[new_cols].fillna(0)
    pos_cash_cols += new_cols
    deleted_cols = pos_cash_remove_low_iv(df)
    pos_cash_cols = list(set(pos_cash_cols) - set(deleted_cols))
    return df, pos_cash_cols
    
    