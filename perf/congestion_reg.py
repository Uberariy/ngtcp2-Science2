"""
This program is an .ipybn notebook utility.

"""

# %%
'''Import libraries'''
from sklearn import tree, preprocessing, model_selection, ensemble, metrics, svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.compose import TransformedTargetRegressor
import openpyxl
import re
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import math
import random
import json
import itertools
import signal
import scipy
from collections import defaultdict
import pickle
from pathlib import Path
# from pure_sklearn.map import convert_estimator

'''We are using this libraries, because they are observed in "SDTR: Soft Decision Tree Regressor for Tabular
Data" article - HAORAN LUO , FAN CHENG , (Member, IEEE), HENG YU , (Graduate Student Member, IEEE), AND YUQI YI , (Student Member, IEEE)'''
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
# from deepforest import CascadeForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso

import time


# %%
'''Define pathes and general settings'''

'''Xlsx file to parse data from'''
xlsx_df_source = "cc_perf_tests/optimal_speed_ML_v2.xlsx"
'''Path, where to save ml cross val results'''
cc_dir = "cc_ml_diploma/"
'''Filenames with ml results'''
ml_result_names = ['']
'''Path to save models'''
path_to_save_ml_models = Path("../stand_support/models/")
sns.set_theme(style="whitegrid")

# %%
'''Import data and put in pd.DataFrame format'''

def importdf2(path):
    '''Import data from an xlsx with all algos in it'''
    wb = openpyxl.load_workbook(path)
    ws = wb["Sheet1"]
    columns = next(ws.values)[0:]
    df = pd.DataFrame(ws.values, columns=columns)
    df = df.iloc[1:, 1:]
    return df

df = importdf2(xlsx_df_source)

'''If somebody asks...'''
df.describe()
# df['Channel Loss (%)'].unique()

# df
# df_estimated_bbr_params

# %%
'''Preprocessing'''
def preprocess(df):
    '''Convert data types'''
    col = list(df.columns)

    '''Remove columns, that we don't need for learning'''
    col.remove('Congestion Controller')
    col.remove('Channel Jitter (ms)')
    col.remove('Estimation iterations number')
    col.remove('Optimal Speed (Kbit/s)')

    '''Float64'''
    d = dict()
    for i in col:
        d[i] = 'float64'

    df = df.astype(d)

    col.remove('Optimal CWND (bytes)')

    X = pd.DataFrame(df, columns=col)
    y = df['Optimal CWND (bytes)']

    '''Try scaler'''
    # scaler = preprocessing.StandardScaler()
    # scaler.fit(X)
    # X = scaler.transform(X)
    return pd.DataFrame(X, columns=col), y

X_df, y_df = preprocess(df)

# %%
'''Split train & test'''
random_state_ = 44

X_cv, X_test, y_cv, y_test = model_selection.train_test_split(X_df, y_df, test_size=0.1, random_state=random_state_, shuffle=True)

X = X_cv
y = y_cv

print(X_cv.shape, X_test.shape, y_cv.shape, y_test.shape)

# %%
'''Clear results'''
res_list = list()

# %%

'''Print cross val results and save them in 'cc_dir' '''
def conclude_and_save(res_list, total_time, lib_name="", finalists_numb=10):
    print(f'\nTotal {total_time} taken.')
    res_list = sorted(res_list, key=lambda x: -x['test_NMAE'])
    print(f"\nTop {finalists_numb} models: ", res_list[:10])
    fname = f'{cc_dir}/ml_cwnd_'+lib_name+'_'+'.'.join('_'.join(time.asctime().split()).split(':'))
    with open(fname, 'w') as f:
        json.dump(res_list, f, indent = 6)
    print("\nName: ", fname)

'''Custom regression metrics'''
def negative_sqrt_mean_squared_error(y_true, y_pred):
    '''This is to apply sqrt and use "greater_is_better=True"'''
    res = metrics.mean_squared_error(y_true, y_pred)
    return - np.sqrt(res)

def negative_mean_absolute_error(y_true, y_pred):
    '''This is to use "greater_is_better=True"'''
    res = metrics.mean_absolute_error(y_true, y_pred)
    return - res

def negative_error_percent(y_true, y_pred):
    '''Custom metric'''
    # print("!!!", y_true, "!!!", y_pred)
    y_true, y_pred = np.sort(np.array(y_true).ravel()), np.sort(np.array(y_pred).ravel())
    # print("!!!", y_true, "!!!", y_pred)
    res = - np.mean(np.abs((y_true - y_pred) / y_true * 100))
    return res

def negative_error_percent_no_abs(y_true, y_pred):
    '''Custom metric. Differs from with abs version in aim to see, wherther we overstate or underestimate'''
    y_true, y_pred = np.sort(np.array(y_true).ravel()), np.sort(np.array(y_pred).ravel())
    res = - np.mean((y_true - y_pred) / y_true * 100)
    return res

'''Cross val'''
def cross_val(X, y, paramGrid, modelF, libName, expConstraint=10000, timeConstraint=60):
    total_time = time.time()
    i = 1
    '''Constraint on holding data, this would be enough'''
    max_paramGrid_size = 1000000
    paramGrid_size = min(math.prod([len(paramGrid[key]) for key in paramGrid.keys()]), max_paramGrid_size)

    iteratedGrid_size = 0
    iteratedGrid_set = []

    class TooLongExperiment(Exception):
        '''Raise if algorithm train is way too long'''
        pass

    def signal_handler(signum, frame):
        raise TooLongExperiment("\nPrevious experiment timed out!")

    '''Put regression metrics into a tuple'''
    metrics_tuple = {
        'NSMSE': metrics.make_scorer(negative_sqrt_mean_squared_error, greater_is_better=True),
        'NMAE': metrics.make_scorer(negative_mean_absolute_error, greater_is_better=True),
        'R2': metrics.make_scorer(metrics.r2_score, greater_is_better=True),
        'NEP': metrics.make_scorer(negative_error_percent, greater_is_better=True),
    }

    '''2 is the unreachable best result for each metric'''
    metrics_val = {
        'test_NSMSE': 2,
        'test_NMAE': 2,
        'test_R2': 2,
        'test_NEP': 2
    }

    '''For current cross val status output'''
    metrics_val_best = {
        'test_NSMSE': 2,
        'test_NMAE': 2,
        'test_R2': 2,
        'test_NEP': 2
    }
    while ((iteratedGrid_size != expConstraint) and (iteratedGrid_size != paramGrid_size // 2)):
        '''We don't want to go through all of the iterations => '... // 2' '''
        ijk = dict()
        for key in paramGrid.keys():
            ijk[key] = random.choice(paramGrid[key])

        if ijk in iteratedGrid_set:
            '''If this experiment is already done, we don't need to repeat'''
            continue

        iteratedGrid_size += 1
        iteratedGrid_set.append(ijk)

        initialtime = time.time()
        model = modelF(ijk)
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(timeConstraint)
        try:
            cv_results = model_selection.cross_validate(model, X, y, cv=model_selection.ShuffleSplit(n_splits=5, test_size=0.25, random_state=0), scoring=metrics_tuple, n_jobs=-1)
        except TooLongExperiment as E:
            print(E)
            signal.alarm(0)
            continue
        except KeyboardInterrupt as E:
            print('\nKeyboardInterrupt => Returning...')
            signal.alarm(0)
            break
        signal.alarm(0)
        for key in metrics_val.keys():
            metrics_val[key] = cv_results[key].mean()
        resulttime = time.time() - initialtime

        '''Experiment Results'''
        print(f'\nExperiment {i}/{min(paramGrid_size // 2, expConstraint)};   Time {resulttime};   Parameters: {ijk}')
        for key in metrics_val.keys():
            if (metrics_val_best[key] < metrics_val[key]) or (metrics_val_best[key] == 2):
                metrics_val_best[key] = metrics_val[key]
            print(f'Score {key[5:]}\t {metrics_val[key]} \t(Best {metrics_val_best[key]})')
        i += 1
        exp_dict = dict()
        exp_dict['library'] = libName
        for key in paramGrid.keys():
            exp_dict[key] = ijk[key]
        exp_dict['training_duration'] = resulttime
        for key in metrics_val.keys():
            exp_dict[key] = metrics_val[key]
        res_list.append(exp_dict)

    return res_list, time.time() - total_time


# %%
'''LightGBM'''
paramGrid = {
    'learning_rate': list(np.random.uniform(0.01, 0.1, 20)) + list(np.random.uniform(0.001, 0.01, 20)) + list(np.random.uniform(0.0001, 0.001, 20)),
    'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    'n_estimators': [int(i) for i in list(np.random.uniform(10, 2000, 50))],
    'num_leaves': [2, 3, 4, 5, 6, 7, 8, 9]
}

def LGBMRegressor_model(ijk):
    return TransformedTargetRegressor(regressor=LGBMRegressor(**ijk),
                                      transformer = preprocessing.MinMaxScaler())

res_list, total_time = cross_val(X, y, paramGrid, LGBMRegressor_model, 'LightGBM', 500, timeConstraint=30)

# %%
conclude_and_save(res_list, total_time, lib_name='LightGBM')
'''Optional: Run   res_list = list()'''

# %%
'''CatBoost'''
paramGrid = {
    'num_trees': list(np.random.uniform(2, 100, 30)) + list(np.random.uniform(100, 1000, 10)) + list(np.random.uniform(1000, 3000, 20)),
    'depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    'l2_leaf_reg': [0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    'silent': [True]
}

def CatBoost_model(ijk):
    return CatBoostRegressor(**ijk)

res_list, total_time = cross_val(X, y, paramGrid, CatBoost_model, 'CatBoost', 500, timeConstraint=30)

# %%
conclude_and_save(res_list, total_time, lib_name='CatBoost')
'''Optional: Run   res_list = list()'''

# %%
'''XGBoost'''
def loguniform(low=0, high=1, size=None):
    return np.exp(np.random.uniform(low, high, size))
paramGrid = {
    'learning_rate': list(np.random.uniform(0.01, 0.1, 20)) + list(np.random.uniform(0.1, 1, 20)),
    'n_estimators': [int(i) for i in list(np.random.uniform(0, 1000, 5)) + list(np.random.uniform(1000, 3000, 5))],
    'max_depth': [int(i) for i in list(np.random.uniform(2, 14, 30))],
    'subsample': list(np.random.uniform(0.5, 1, 20)),
    'colsample_bytree': list(np.random.uniform(0.5, 1, 20)),
    'colsample_bylevel': list(np.random.uniform(0.5, 1, 20)),
    'min_child_weight': list(np.random.uniform(1, 7, 20)),
    'reg_alpha': list(np.random.uniform(0, 12, 20)) + [0],
    'reg_lambda': list(np.random.uniform(0, 12, 20)) + [0],
    'gamma': list(np.random.uniform(0.5, 1, 20)) + [0]
}

def XGBoost_model(ijk):
    return TransformedTargetRegressor(regressor=XGBRegressor(**ijk),
                                      transformer = preprocessing.MinMaxScaler())

res_list, total_time = cross_val(X, y, paramGrid, XGBoost_model, 'XGBoost', 200, timeConstraint=30)

# %%
conclude_and_save(res_list, total_time, lib_name='XGBoost')
'''Optional: Run   res_list = list()'''
res_list = list()

# %%
'''Sklearn Tree'''
paramGrid = {
    'n_estimators': [int(i) for i in list(np.random.uniform(1, 650, 50))],
    'max_depth': [11, 12, 13, 14, 15, 16, 17],
    'max_features': [1.0, 'sqrt'],
    'min_samples_leaf': [1, 2, 4, 8],
    'min_samples_split': [2, 5, 10]
}

def Sklearn_model(ijk):
    return TransformedTargetRegressor(regressor=ensemble.RandomForestRegressor(**ijk),
                                      transformer=preprocessing.MinMaxScaler())

res_list, total_time = cross_val(X, y, paramGrid, Sklearn_model, 'Sklearn', 400, timeConstraint=30)

# %%
conclude_and_save(res_list, total_time, lib_name='Sklearn')
'''Optional: Run   res_list = list()'''
res_list = list()

# %%
'''Sklearn Lasso'''
paramGrid = {
    'alpha': [0.05, 0.1, 0.5, 1.0, 10.0, 100.0, 1000.0],
    'selection': ['cyclic', 'random']
}

def Sklearn_Lasso_model(ijk):
    return Lasso(**ijk)

res_list, total_time = cross_val(X, y, paramGrid, Sklearn_Lasso_model, 'Sklearn Lasso')

# %%
# conclude_and_save(res_list, total_time)
res_list = list()
'''Optional: Run   res_list = list()'''

# %%
'''Sklearn Ridge'''
paramGrid = {
    'alpha': [0.05, 0.1, 0.5, 1.0, 10.0, 100.0, 1000.0],
    'solver': ['auto', 'svd', 'cholesky', 'sparse_cg', 'lsqr', 'sag']
}

def Sklearn_Ridge_model(ijk):
    return Ridge(**ijk)

res_list, total_time = cross_val(X, y, paramGrid, Sklearn_Ridge_model, 'Sklearn Ridge')

# %%
conclude_and_save(res_list, total_time)
'''Optional: Run   res_list = list()'''

# %%
'''Sklearn SVR'''
paramGrid = {
    'kernel': ['poly'] + ['rbf', 'sigmoid', 'linear'],
    'degree': list(np.random.uniform(1, 15, 30)),
    'epsilon': [0.1] + list(np.random.uniform(0.1, 1, 20)) + list(np.random.uniform(1, 10, 20)) + list(np.random.uniform(10, 100, 3)),
    'C': [1] + list(np.random.uniform(0.1, 1, 20)) + list(np.random.uniform(1, 10, 20)) + list(np.random.uniform(10, 100, 3)),
    'gamma': 10*['auto'] + 5*['scale'] + list(np.random.uniform(0.1, 1, 5)),
    'coef0': 20*[0.0] + list(np.random.uniform(0.1, 1, 20)) + list(np.random.uniform(1, 10, 20)) + list(np.random.uniform(10, 100, 3)),
    'shrinking': 8*[True] + [False]
}

def Sklearn_SVR(ijk):
    return make_pipeline(StandardScaler(), svm.SVR(**ijk))

res_list, total_time = cross_val(X, y2, paramGrid, Sklearn_SVR, 'Sklearn SVR y2', timeConstraint=20)

# %%
res_list = list()
conclude_and_save(res_list, total_time)
'''Optional: Run   res_list = list()'''

# %%
'''Invsee Cross Validation progress, build graphs to better understand parameter samples quality.
Put the files with ML cross val results in 'ml_result_names' list
The parameters chosen and the metrics must be same for all the experiments!'''

def plot_cross_val_rating(cc_dir, ml_result_names, main_metric="test_R2"):
    results = []
    for filename in ml_result_names:
        with open(cc_dir + '/' + filename, 'r') as f:
            results += json.load(f)

    '''Pick a certain amount of best models, to place on a graph'''
    best_models_amount = 50
    results = results[:best_models_amount]
    n = 2
    m = 3
    fig, axes = plt.subplots(n, m, figsize=(17, 13))
    fig.suptitle('Dependencies between chosen parameters and accuracy score.')

    k = 0
    for i in range(n):
        for j in range(m):
            k += 1
            if ((k >= len(results[0].keys())) or (list(results[0].keys())[k] == "gamma")):
                axes[i, j].set_title('None')
                continue
            axes[i, j].set_title('Parameter ' + list(results[0].keys())[k])
            x_ = [p[list(results[0].keys())[k]] for p in results]
            for idx, u in enumerate(x_):
                if isinstance(u, str):
                    x_[idx] = 0
            y_ = [p[main_metric] for p in results]
            x_name = "Parameter value"
            y_name = "Accuracy metric " + main_metric
            sns.scatterplot(ax=axes[i, j], data=pd.DataFrame({x_name: x_, y_name: y_}), x=x_name, y=y_name)

plot_cross_val_rating(cc_dir, ['ml_cwnd_Sklearn_Tue_Mar_28_15.31.40_2023'])

# %%
'''Correlation betweeen chosen metrics'''

def plot_metrics_correlation(cc_dir, ml_result_names):
    results = []
    for filename in ml_result_names:
        with open(cc_dir + '/' + filename, 'r') as f:
            results += json.load(f)

    best_models_amount = 80
    results = results[:best_models_amount]

    fig, axes = plt.subplots(4, 4, figsize=(28, 28))
    fig.suptitle('Dependencies between chosen parameters and accuracy score.')

    metrics_names = ['test_NSMSE', 'test_NMAE', 'test_R2', 'test_NEP']
    for i in range(4):
        for j in range(4):
            axes[i, j].set_title(f'Dependency {metrics_names[i]} by {metrics_names[j]}')
            x_ = [p[metrics_names[j]] for p in results]
            y_ = [p[metrics_names[i]] for p in results]
            sns.lineplot(ax=axes[i, j], data=pd.DataFrame({f"Metric {metrics_names[i]}": x_, f"Metric {metrics_names[j]}": y_}), x=f"Metric {metrics_names[i]}", y=f"Metric {metrics_names[j]}", ci=100)

plot_metrics_correlation(cc_dir, ml_result_names)

# %%
'''Functions to experiment with models after crossvalidation finished.'''

'''Train & Test'''
def manual_train_test(model, X_train, y_train, X_test, y_test, metrics=[metrics.r2_score, negative_error_percent]):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_score = []
    for metric in metrics:
        test_score.append(metric(y_test, y_pred))
    return y_pred, test_score

'''For next function'''
target_type_str = [
    'Sender Speed (Kbit/s)',
    'MinimalSpeed',
    'MaximalSpeed'
]
alpha = 1

'''Examine if results differ much'''
def manual_invsee_results(target_str, y_test, y_pred, metrics={"Negative error percent:": negative_error_percent, "The bias:": negative_error_percent_no_abs, "R2 score:": metrics.r2_score}, silent=False):
    results = y_test.to_frame().assign(Predicted=y_pred)
    # sorted_results = y_test.to_frame().assign(Predicted=y_pred).sort_values(by=target_str)
    results['Abs Error (%)'] = (results[target_str] - results['Predicted']) / results[target_str] * 100
    # print("BIAS: |", results['Abs Error (%)'].mean(), "|")
    results['Squared Error'] = (results[target_str] - results['Predicted']) ** 2 / results[target_str] ** 2 * 100
    results.rename({'Predicted': 'Predicted value (Kbit/s)'}, inplace=True)
    if not silent:
        print("Metrics:") 
    metric_results = {key: metric(y_test, y_pred) for key, metric in metrics.items()}
    bias, nep_metric = None, None
    for key, val in metric_results.items():
        if not silent:
            print(key, val)
        if key == "Negative error percent:":
            nep_metric = val
        elif key == "The bias:":
            bias = val
    if not silent:
        print("\n", results.tail(10))
    return results, bias, nep_metric # Return bias to calc

'''Try to ensemble two plots, and see how they combine'''
def model_2_ensemble_plot(y_pred1, y_pred2, y_test, metrics=[metrics.r2_score, negative_error_percent]):
    alpha_min = 0
    alpha_max = 1
    ls = np.linspace(alpha_min, alpha_max, 101)
    quals = []
    for metric in metrics:
        quals.append([metric(y_test, y_pred1 * i + (1 - i) * y_pred2) for i in ls])
    plt.figure(figsize=(8, 8 * len(metrics)))
    fig, axes = plt.subplots(1, len(metrics), figsize=(10, 5))
    fig.suptitle('Changing alpha parameter to see, how models impact in overall result.')
    for idx, (qual, metric) in enumerate(zip(quals, metrics)):
        axes[idx].set_title('Metric: ' + str(metric.__name__))
        axes[idx].plot(ls, qual, lw=2)
        axes[idx].set_xlabel('Alpha')
        axes[idx].set_ylabel('Quality')
        axes[idx].set(xlim=[alpha_min, alpha_max])
    for qual in quals:
        print('Max dual metric: ', max(qual), '\nAlpha: ', ls[qual.index(max(qual))])

'''Try to ensemble two plots, and see how they combine'''
def model_3_ensemble_plot(y_pred1, y_pred2, y_pred3, y_test, metrics=[metrics.r2_score, negative_error_percent]):
    alpha_min = 0
    alpha_max = 1
    size = 20
    quals = []
    max_quals = []
    first_quals = []
    alphas = [0 for i in range(len(metrics))]
    betas = [0 for i in range(len(metrics))]
    for metric in metrics:
        max_quals.append(metric(y_test, y_pred1))
        first_quals.append(metric(y_test, y_pred1))
    ls = []
    for idx, metric in enumerate(metrics):
        sup_quals = []
        for k, j in enumerate(np.linspace(alpha_min, alpha_max, size + 1)):
            sub_quals = []
            sub_ls = []
            for i in np.linspace(alpha_min, alpha_max - j, size + 1 - k):
                qual = metric(y_test, y_pred1 * j + y_pred2 * i + y_pred3 * (1 - i - j))
                if (max_quals[idx] < qual):
                    max_quals[idx] = qual
                    alphas[idx] = j
                    betas[idx] = i
                sub_quals.append(qual)
                sub_ls.append(i)
            for i in np.linspace(alpha_max - j, alpha_max, k):
                sub_quals.append(-200)
            sup_quals.append(sub_quals)
            ls.append(sub_ls)
        quals.append(sup_quals)
    # print(quals[0])
    plt.figure(figsize=(8, 12 * len(metrics)))
    fig, axes = plt.subplots(1, len(metrics), figsize=(10, 5))
    fig.suptitle('Changing alpha, beta parameter to see, how models impact in overall result.')
    for idx, qual in enumerate(quals):
        # print(ls, qual)
        sns.heatmap(qual, ax=axes[idx], vmin=min(qual[0][0]/(100/99), qual[0][0]*(7/6)))
        axes[idx].set_title('Metric: ' + str(metrics[idx].__name__))
        axes[idx].set_xlabel('Beta')
        axes[idx].set_ylabel('Alpha')
        # axes[idx].set_xticks(np.linspace(alpha_min, alpha_max, size + 1, 10))
        # axes[idx].set_yticks(np.linspace(alpha_min, alpha_max, size + 1, 10))
        # axes[idx].set(xlim=[alpha_min, alpha_max])
    print('Max dual metrics: ', max_quals, '\nAlpha: ', alphas, '\nBeta: ', betas)

'''Do we want to handle more experiments for new input samples?'''
def train_size_plot(model, X, y, metric=negative_error_percent):
    iterations = 20
    iterations_for_each = 90
    ls = np.linspace(0.3, 0.93, iterations)
    sizes = []
    quals = []
    for idx, i in enumerate(ls):
        quals_tmp = []
        for j in range(iterations_for_each):
            X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=1-i, shuffle=True)
            _, cv_score = manual_train_test(model, X_train, y_train, X_test, y_test, metrics=[metric])
            quals_tmp.append(cv_score[0])
        quals.append(np.mean(quals_tmp))
        sizes.append(X_train.shape[0])
        print("train_size_plot: [", idx+1, " / ", iterations, "]")
    plt.figure(figsize=(10, 10))
    plt.plot(sizes, quals, lw=2)
    plt.xlabel('Train size')
    plt.ylabel('Quality')

'''Try using stacking - Current results are worse than models themselves'''
def stacking_attempt(model1, model2):
    '''This function uses globals'''
    estimators = [
        ('1', model1),
        ('2', model2)
    ]
    stacking = ensemble.StackingRegressor(
        estimators=estimators,
        final_estimator=ensemble.RandomForestRegressor(n_estimators=2000, max_depth=8)
    )

    y_pred0, cv_score0 = manual_train_test(stacking, X, y1, X_test, y1_test)
    results = manual_invsee_results(target_type_str[0], y1_test, y_pred0)

'''Show some metrics correllation'''
def plot_features_and_target_correlation(X_df, y1_df, y2_df, y3_df):
    plt.imshow(pd.concat([X_df, y1_df, y2_df, y3_df], axis=1).corr(method='kendall').apply(lambda x: np.abs(x)))
    pd.concat([X_df, y1_df, y2_df, y3_df], axis=1).corr(method='kendall')

'''Just a plot to invsee results.
Plot in the next function is customizable depending on the constraints set'''
def custom_test_train_plot(X, y1_test, y2_test, y3_test, y1_pred, y2_pred, y3_pred):
    '''Cusom constraints:'''
    X = X.loc[(X['Congestion Controller'] == 3)]
    X = X.loc[(X['Channel BW (Kbit/s)'] >= 50000) & (X['Channel BW (Kbit/s)'] >= 250000)]
    X = X.loc[(X['Channel RTT (ms)'] >= 50) & (X['Channel RTT (ms)'] >= 0)]
    X = X.loc[(X['Channel Loss (%)'] >= 1) & (X['Channel Loss (%)'] >= 0)]
    x_feature = 'Channel BW (Kbit/s)'

    '''Make a plot'''
    X = X.sort_values(by=x_feature, ascending=False)
    idx = list(X.index.values)
    plt.figure(figsize=(13, 10))
    X_axis = X[x_feature]

    # X_axis = str(X_axis[x_feature])

    plt.plot(X_axis, y1_test[idx], label='Speed (Kbit/s)', color='#85F1B8') # 85F1B8
    plt.plot(X_axis, pd.Series(y1_pred, index=y1_test.index)[idx], label='Predicted Mean Speed (Kbit/s)', color='#0DD26A')
    plt.plot(X_axis, y2_test[idx], label='Real Min Speed (Kbit/s)', color='#F4AE7B')
    plt.plot(X_axis, pd.Series(y2_pred, index=y2_test.index)[idx], label='Predicted Min Speed (Kbit/s)', color='#DE630B')
    plt.plot(X_axis, y3_test[idx], label='Real Max Speed (Kbit/s)', color='#7BB6F4')
    plt.plot(X_axis, pd.Series(y3_pred, index=y3_test.index)[idx], label='Predicted Max Speed (Kbit/s)', color='#146CCA')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel(x_feature)
    plt.ylabel('Sender Speed (Kbit/s)')
    plt.legend()

'''Final ensemble'''
def net_speed_predict_ensemble(best_models, X_test, measuretime=False):
    channels = X_test #.head(20)
    predictions = dict()
    if measuretime:
        start_time = time.time()
    for key, model in best_models.items():
        predictions[key] = model.predict(channels)
    '''Emperical alpha/beta below:'''
    predictions[1] = predictions[11] * (0.85) + predictions[12] * (0.0) + predictions[13] * (0.15)
    predictions[2] = predictions[21] * (0.25) + predictions[22] * (0.5) + predictions[23] * (0.25)
    predictions[3] = predictions[31] * (0.5) + predictions[32] * (0.3) + predictions[33] * (0.2)
    # model_3_ensemble_plot(predictions[1], predictions[2], predictions[3], y1_test)
    y_pred = predictions[1] * (0.85) + predictions[2] * (0.1) + predictions[3] * (0.05)
    if measuretime:
        print("Time taken for predict in seconds:", time.time() - start_time)
    return y_pred

'''Measure predict time. '''
def measure_time_best_models_no_ensemble(best_models, X_test):
    times = []
    for i in range(10):
        start_time = time.time()
        for key, model in best_models.items():
            _ = model.predict(X_test.head(20))
        times.append(time.time() - start_time)
    print(sum(times) / len(times))
    '''Let 0.3s. be a nice threshhold'''

'''Having estimated best BBR parameters, and speed, corresponding them, compare it with speed on BBR with standard parameters, that we get from pretrained ML model'''
def compare_estimated_best_params_speed_and_ml_speed_on_standard_params(trusted_model, estimated_bbr_params_df, ensemble=True):
    cols = ["Congestion Controller", "Channel RTT (ms)", "Channel Loss (%)", "Channel BW (Kbit/s)"]
    
    df = estimated_bbr_params_df[cols]
    '''Float64'''
    d = dict()
    for i in cols:
        d[i] = 'float64'
    '''Categorial'''
    d['Congestion Controller'] = 'str'
    df = df.astype(d)
    df["Congestion Controller"] = df["Congestion Controller"].apply(lambda x: list(df["Congestion Controller"].unique()).index(x) + 1)
    pd.to_numeric(df['Congestion Controller'])

    if ensemble:
        predictions = net_speed_predict_ensemble(trusted_model, df, measuretime=False)
    else:
        predictions = trusted_model.predict(df)
    results = estimated_bbr_params_df
    results['Base speed predicted by a models (Kbit/s)'] = predictions
    results['Speed profit when using custom BBR params (%)'] = (results['Optimal Speed (Kbit/s)'] - results['Base speed predicted by a models (Kbit/s)']) / results['Optimal Speed (Kbit/s)'] * 100
    print("Mean speed profit in percent (%) is ", - negative_error_percent_no_abs(results['Optimal Speed (Kbit/s)'], results['Base speed predicted by a models (Kbit/s)']))
    return results

'''Preprocess data for BBR Parameters calculation'''
def preprocess_bbr_parameters(trusted_models, estimated_bbr_params_df, ensemble=True):
    df = compare_estimated_best_params_speed_and_ml_speed_on_standard_params(trusted_models, estimated_bbr_params_df, ensemble)

    df['BBR Beta'] = np.where(df['Speed profit when using custom BBR params (%)'] < 0.25, 0.7, df['BBR Beta'])
    df['BBR Losstresh'] = np.where(df['Speed profit when using custom BBR params (%)'] < 0.25, 0.05, df['BBR Losstresh'])
    df['BBR Probe RTT Cwnd Gain'] = np.where(df['Speed profit when using custom BBR params (%)'] < 0.25, 0.5, df['BBR Probe RTT Cwnd Gain'])
    df['BBR Probe RTT Duration'] = np.where(df['Speed profit when using custom BBR params (%)'] < 0.25, df['Channel RTT (ms)'] * 2, df['BBR Probe RTT Duration'])
    
    W_df = pd.DataFrame(df, columns = ["Channel RTT (ms)", 'Channel Loss (%)', "Channel BW (Kbit/s)", "Base speed predicted by a models (Kbit/s)"]).astype('float64')
    u1_df = df["BBR Beta"]
    u2_df = df["BBR Losstresh"]
    u3_df = df["BBR Probe RTT Cwnd Gain"]
    u4_df = df["BBR Probe RTT Duration"]
    return W_df, u1_df, u2_df, u3_df, u4_df

def X_poly(X, degree_const = 3):
    polynomial_features = PolynomialFeatures(degree=degree_const)
    res = polynomial_features.fit_transform(X)
    # print(polynomial_features.powers_)
    return res

# %%
'''Best model experiments and bias by multiple runs'''
biases = []
neps = []

runs_number = 1

for i in range(runs_number):
    X_cv, X_test, y_cv, y_test = model_selection.train_test_split(X_df, y_df, test_size=0.1, shuffle=True)

    best_model = Lasso(alpha = 20000, selection = 'random', tol = 3*1e-5, max_iter = 160000, warm_start = True, precompute = True) # 0.15397035347669225 -2.078726559288437
    # best_model = Ridge(alpha = 50, solver='cholesky') # -0.1247498447031332 -1.9871304715218807
    degree = 7

    X_cv_poly = X_poly(X, degree_const=degree)
    X_test_poly = X_poly(X_test, degree_const=degree)

    test_pred, test_score = manual_train_test(best_model, X_cv_poly, y, X_test_poly, y_test)
    _, cur_nep, cur_bias = manual_invsee_results("Optimal CWND (bytes)", y_test, test_pred, {"Negative error percent:": negative_error_percent, "The bias:": negative_error_percent_no_abs, "R2 score:": metrics.r2_score}, silent=True)
    
    biases.append(cur_bias)
    neps.append(cur_nep)

print("\n", sum(neps)/len(neps), sum(biases)/len(biases))

# %%
'''Save best models'''
pickle.dump(best_model, open(path_to_save_ml_models / "poly_reg_lasso_degree_deg7.txt", 'wb'))


# %%
'''Do we want to handle more experiments for new input samples?'''
if (True):
    train_size_plot(TransformedTargetRegressor(regressor=LGBMRegressor(),
                                            transformer=preprocessing.MinMaxScaler()),
                                            X, y, negative_error_percent) 

# %%
'''Try...'''
def X_custom(X, degree_const=3):
    def f1(a):
        return 1 / (a + 1)
    X_poly_tmp = X_poly(np.concatenate((X, np.vectorize(f1)(X)), axis=1), degree_const)
    functions = [
        np.log,
    ]
    return np.concatenate((X_poly_tmp, *[f(X_poly_tmp) for f in functions]), axis=1)

'''Poly'''
def try_regressor(X, y, X_test, y_test, degree):
    X_cv_poly = X_poly(X, degree_const=degree)
    X_test_poly = X_poly(X_test, degree_const=degree)

    clf = Lasso(alpha = 17000, selection = 'random', tol = 3*1e-4, max_iter = 30000, warm_start = True, precompute = True) # 
    # clf = Lasso(alpha = 3908, selection = 'random', tol = 3*1e-5, max_iter = 520000, warm_start = True, precompute = True) # -3.973408981892189
    # clf = Ridge(alpha = 0.07, tol = 10000, max_iter = 10200, solver='cholesky') # -2.3982246154979414, degree=7
    clf = clf.fit(X_cv_poly, y)

    '''Trying to remove small weights - does help'''
    # clf.coef_[(clf.coef_*clf.coef_ < 5e8)] = 0 

    '''See coefficients'''
    print(clf.coef_)

    '''MSE Results'''
    y_pred = clf.predict(X_test_poly)
    nep = negative_error_percent(y_test, y_pred)
    print(nep, "\n", pd.DataFrame({"test": y_test, "pred": y_pred}))
    print("Features:", len(clf.coef_), "Nonzero:", np.count_nonzero(clf.coef_))
    return nep

'''Try'''
# try_regressor(X[['Channel RTT (ms)', 'Channel BW (Kbit/s)']], y, X_test[['Channel RTT (ms)', 'Channel BW (Kbit/s)']], y_test, degree=4)
try_regressor(X, y, X_test, y_test, degree=7)

# %%
X

# %%
'''Try classic approaches'''
def BDP(X):
    return pd.DataFrame(X['Channel BW (Kbit/s)'] * X['Channel RTT (ms)'] / 8, columns=["BDP"])

def BBR_FORECAST_2022(X):
    return pd.DataFrame(X['Channel BW (Kbit/s)'] * X['Channel RTT (ms)'] / 8 * 100 / (100 + X['Channel Loss (%)'] / (100 - X['Channel Loss (%)'])))

biases = []
neps = []

'''Best model experiments and bias by multiple runs'''
runs_number = 40
for i in range(runs_number):
    X_cv, X_test, y_cv, y_test = model_selection.train_test_split(X_df, y_df, test_size=0.1, shuffle=True)

    # y_pred_tmp = manual_train_test(Ridge(), BDP(X), y, BDP(X_test), y_test) # 5.199098635928246 -8.845153115373808
    # _, cur_nep, cur_bias = manual_invsee_results("Optimal CWND (bytes)", y_test, y_pred_tmp[0])
    _, cur_nep, cur_bias = manual_invsee_results("Optimal CWND (bytes)", y_test, BDP(X_test), silent=True) # -2.8089529091172727 -6.20016727310029

    # y_pred_tmp = manual_train_test(Ridge(), BBR_FORECAST_2022(X), y, BBR_FORECAST_2022(X_test), y_test) # 5.785538055330505 -9.583897965352309
    # _, cur_nep, cur_bias = manual_invsee_results("Optimal CWND (bytes)", y_test, y_pred_tmp[0])
    # _, cur_nep, cur_bias = manual_invsee_results("Optimal CWND (bytes)", y_test, BBR_FORECAST_2022(X_test)) # -3.0237179236415557 -5.928357987302471

    biases.append(cur_bias)
    neps.append(cur_nep)

print("\n", sum(neps)/len(neps), sum(biases)/len(biases))

# %%
degree=3, alpha = 3908
[     0.         447643.34972128      0.         365321.910971
      0.              0.         121873.5146077       0.
      0.         -41989.40600171      0.              0.
      0.              0.              0.         -38908.03063347
      0.              0.              0.              0.        ]
-5.7729229443728345 
           test          pred
526  1724334.0  1.713797e+06
358   490332.0  5.198735e+05
445   873535.0  8.996284e+05
32    288769.0  3.444303e+05
619   847167.0  8.794671e+05
..         ...           ...
483   896386.0  9.223765e+05
306  2175322.0  2.028146e+06
895   209667.0  1.823761e+05
108   565332.0  6.838027e+05
882   297558.0  2.905127e+05

[103 rows x 2 columns]
Features: 20 Nonzero: 5


# %%
print(X_test, y_test)

# %%
negative_error_percent_no_abs([100, 100], [90, 90])

# %%
