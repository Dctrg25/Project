import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

from sklearn.model_selection import KFold,StratifiedKFold,train_test_split
from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb
import gc
from sklearn.metrics import roc_auc_score

def df_train_test(df):
    #Remove feature present where the client lives
    del_col = []
    for i in df.columns:
        if i.find('_AVG') != -1:
            del_col.append(i)
        if i.find('_MODE') != -1:
            del_col.append(i)
        if i.find('_MEDI') != -1:
            del_col.append(i)
    df.drop(columns = del_col , inplace=True)
     
    try:
        #Remove some rows with values not present in test set
        df.drop(df[df['CODE_GENDER'] == 'XNA'].index, inplace=True)
        df.drop(df[df['NAME_INCOME_TYPE'] == 'Maternity leave'].index, inplace=True)
        df.drop(df[df['NAME_FAMILY_STATUS'] == 'Unknown'].index, inplace=True)
    except:
        print('....')
    #Replace some outliers
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    df.loc[df['OWN_CAR_AGE'] > 80, 'OWN_CAR_AGE'] = np.nan
    df.loc[df['REGION_RATING_CLIENT_W_CITY'] < 0, 'REGION_RATING_CLIENT_W_CITY'] = np.nan
    df.loc[df['AMT_INCOME_TOTAL'] > 1e8, 'AMT_INCOME_TOTAL'] = np.nan
    df.loc[df['AMT_REQ_CREDIT_BUREAU_QRT'] > 80, 'AMT_REQ_CREDIT_BUREAU_QRT'] = np.nan
    df.loc[df['OBS_30_CNT_SOCIAL_CIRCLE'] > 80, 'OBS_30_CNT_SOCIAL_CIRCLE'] = np.nan

    #Feature Engineering
    income_by_organi = df[['AMT_INCOME_TOTAL','ORGANIZATION_TYPE']].groupby('ORGANIZATION_TYPE').median()['AMT_INCOME_TOTAL']
    income_by_occupa = df[['AMT_INCOME_TOTAL','OCCUPATION_TYPE']].groupby('OCCUPATION_TYPE').median()['AMT_INCOME_TOTAL']

    df['AGE'] = df['DAYS_BIRTH'] / -365
    df.loc[df['DAYS_EMPLOYED'] == 365243, 'DAYS_EMPLOYED'] = np.nan
    df['YEAR_EMPLOYED'] = df['DAYS_EMPLOYED'] / -365
    df['AMT_CREDIT / AMT_ANNUITY'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['AMT_INCOME_TOTAL / 12 - AMT_ANNUITY'] = df['AMT_INCOME_TOTAL'] / 12 - df['AMT_ANNUITY']
    df['AMT_INCOME_TOTAL / AMT_ANNUITY'] = df['AMT_INCOME_TOTAL'] / df['AMT_ANNUITY']
    df['AMT_INCOME_TOTAL - AMT_GOODS_PRICE'] = df['AMT_INCOME_TOTAL'] - df['AMT_GOODS_PRICE']
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    df['NEW_INCOME_BY_ORG'] = df['ORGANIZATION_TYPE'].map(income_by_organi)
    df['NEW_INCOME_BY_OCC'] = df['OCCUPATION_TYPE'].map(income_by_occupa)
    df['NEW_EXT_SOURCES_MUL'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['NEW_EXT_SOURCES_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    df['AMT_INCOME_TOTAL'] = np.log1p(df['AMT_INCOME_TOTAL'])
    df['AMT_CREDIT'] = np.log1p(df['AMT_CREDIT'])

    df.drop(columns = ['ORGANIZATION_TYPE','OCCUPATION_TYPE'],inplace = True)

    #Label encoding
    le = LabelEncoder()
    le_count = 0

    for col in df:
        if df[col].dtype == 'object' :
            if len(list(df[col].unique())) <= 2:
                le.fit(df[col])
                df[col] = le.transform(df[col])
                le_count += 1
    print('%d columns were label encoded.' % le_count)

    #One-hot encoding
    categorical_feature = [col for col in df.columns if df[col].dtype == 'object']

    df = pd.get_dummies(df, columns = categorical_feature)

    #Process missing value
    categorical_list = []
    numerical_list = []
    for i in df.columns.tolist():
        if df[i].dtype == 'object':
            categorical_list.append(i)
        else:
            if i == 'TARGET':
                continue
            else:
                numerical_list.append(i)

    Imputer = SimpleImputer(missing_values= np.nan, strategy='mean')
    df[numerical_list] = Imputer.fit_transform(df[numerical_list])

    #Print total missing value
    total = df.isnull().sum().sort_values(ascending = False)
    percent = (df.isnull().sum() / df.isnull().count() * 100).sort_values(ascending = False)
    missing_df = pd.concat([total, percent], axis= 1, keys= ['Total','Percent'])
    missing_df.head(10)
    df = df.reset_index()
    return df,categorical_list

def model(features, test_features,cate_list, n_folds = 5):
    #Extract the ids
    train_ids = features['SK_ID_CURR']
    test_ids = test_features['SK_ID_CURR']

    #Extract the labels for training
    labels = features['TARGET']

    #Remove the ids and target
    features = features.drop(columns = ['SK_ID_CURR','TARGET'])
    test_features = test_features.drop(columns = ['SK_ID_CURR'])

    cat_indices = cate_list
    print('Training Data Shape: ', features.shape)
    print('Testing Data Shape: ', test_features.shape)

    #Extract feature name
    feature_names = list(features.columns)

    #Convert to np arrays
    features = np.array(features)
    test_features = np.array(test_features)

    #Creat the KFold object
    k_fold = StratifiedKFold(n_splits= n_folds, shuffle= True, random_state= 50)

    #Empty array for feature importances
    feature_importance_values = np.zeros(len(feature_names))

    #Empty array for test predictions
    test_predictions = np.zeros(test_features.shape[0])
    
    out_of_fold = np.zeros(features.shape[0])

    valid_scores = []
    train_scores = []

    # Iterate through each fold
    for train_indices, valid_indices in k_fold.split(features,labels):
        
        # Training data for the fold
        train_features, train_labels = features[train_indices], labels[train_indices]
        # Validation data for the fold
        valid_features, valid_labels = features[valid_indices], labels[valid_indices]
        
        # Create the model
        model = lgb.LGBMClassifier(nthread=4, 
            n_estimators=10000, 
            learning_rate=0.02, 
            num_leaves=34, 
            colsample_bytree=0.9497036, 
            subsample=0.8715623, 
            max_depth=8, 
            reg_alpha=0.041545473, 
            reg_lambda=0.0735294, 
            min_split_gain=0.0222415, 
            min_child_weight=39.3259775, 
            silent=-1, 
            verbose=-1,)
        
        # Train the model
        model.fit(train_features, train_labels, eval_metric = 'auc',
                  eval_set = [(valid_features, valid_labels), (train_features, train_labels)],
                  eval_names = ['valid', 'train'], categorical_feature = cat_indices,
                  early_stopping_rounds = 200, verbose = 200)
        
        # Record the best iteration
        best_iteration = model.best_iteration_
        
        # Record the feature importances
        feature_importance_values += model.feature_importances_ / k_fold.n_splits
        
        # Make predictions
        test_predictions += model.predict_proba(test_features, num_iteration = best_iteration)[:, 1] / k_fold.n_splits
        
        # Record the out of fold predictions
        out_of_fold[valid_indices] = model.predict_proba(valid_features, num_iteration = best_iteration)[:, 1]
        
        # Record the best score
        valid_score = model.best_score_['valid']['auc']
        train_score = model.best_score_['train']['auc']
        
        valid_scores.append(valid_score)
        train_scores.append(train_score)
        
        # Clean up memory
        gc.enable()
        del model, train_features, valid_features , train_labels,valid_labels
        gc.collect()
        
    # Make the submission dataframe
    submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': test_predictions})
    
    # Make the feature importance dataframe
    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})
    
    # Overall validation score
    valid_auc = roc_auc_score(labels, out_of_fold)
    
    # Add the overall scores to the metrics
    valid_scores.append(valid_auc)
    train_scores.append(np.mean(train_scores))
    
    # Needed for creating dataframe of validation scores
    fold_names = list(range(n_folds))
    fold_names.append('overall')
    
    # Dataframe of validation scores
    metrics = pd.DataFrame({'fold': fold_names,
                            'train': train_scores,
                            'valid': valid_scores}) 
    
    return submission, feature_importances, metrics

train = pd.read_csv('application_train.csv')
test = pd.read_csv('application_test.csv')

df_train,cate_feature = df_train_test(train)
df_test,cate_feature = df_train_test(test)

import re
df_train = df_train.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
df_test = df_test.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

submission, feature_importances, metrics = model(df_train, df_test, cate_feature)

submission['SK_ID_CURR'] = submission['SK_ID_CURR'].astype('int32')
submission.to_csv('selected_features_submission_py.csv', index = False)