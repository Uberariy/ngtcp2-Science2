import pickle
import pandas as pd
import numpy as np

from sklearn import ensemble
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.compose import TransformedTargetRegressor

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from argparse import ArgumentParser
from pathlib import Path

def X_poly(X, degree_const = 3):
    polynomial_features = PolynomialFeatures(degree=degree_const)
    res = polynomial_features.fit_transform(X)
    return res

def load_and_predict(path_to_models, rtt, loss, bw):
    '''
    rtt - ms,
    loss - %,
    bw - Kbit/s
    '''
    model = pickle.load(open(path_to_models / "poly_reg_lasso_degree_deg7.txt", 'rb'))
    degree = 7

    col = ["Channel RTT (ms)", "Channel Loss (%)", "Channel BW (Kbit/s)"]
    X = X_poly(pd.DataFrame([[rtt, loss, bw],], columns=col), degree_const=degree)

    y_pred = model.predict(X)

    print(X, y_pred)

    return y_pred[0] * 0.9985

arg_parser = ArgumentParser(prog='choose_channel',
                            description='')   
arg_parser.add_argument('path_models', metavar='PATH_MODELS', nargs='?', type=Path, default='models/',
                        help='Path where the models used are. ')

arg_parser.add_argument('rtt', metavar='RTT', type=int,
                        help='Rtt (ms) for dataset. ')
arg_parser.add_argument('loss', metavar='LOSS', type=float,
                        help='Loss (%) for dataset. ')
arg_parser.add_argument('bw', metavar='BW', type=int,
                        help='Bw (Kbit/s) for dataset. ')

if __name__ == "__main__":
    '''Change config if needed: '''
    path_to_models = Path("models/")

    args = arg_parser.parse_args()
    result_vector = load_and_predict(args.path_models, args.rtt, args.loss, args.bw)

