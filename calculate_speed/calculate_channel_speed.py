import pickle
import pandas as pd
import numpy as np

from sklearn import ensemble
from sklearn.compose import TransformedTargetRegressor

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from argparse import ArgumentParser
from pathlib import Path

def load_and_predict(path_to_models, cc, rtt, loss, bw):
    models = dict()

    models[11] = XGBRegressor()
    models[21] = XGBRegressor()
    models[31] = XGBRegressor()
    models[11].load_model(Path(path_to_models) / "xgboost_y1.txt")
    models[21].load_model(Path(path_to_models) / "xgboost_y2.txt")
    models[31].load_model(Path(path_to_models) / "xgboost_y3.txt")

    models[22] = pickle.load(open(Path(path_to_models) / "random_forest_y2.txt", 'rb'))
    models[32] = pickle.load(open(Path(path_to_models) / "random_forest_y3.txt", 'rb'))

    models[13] = CatBoostRegressor()
    models[23] = CatBoostRegressor()
    models[33] = CatBoostRegressor()
    models[13].load_model(Path(path_to_models) / "catboost_y1.cbm")
    models[23].load_model(Path(path_to_models) / "catboost_y2.cbm")
    models[33].load_model(Path(path_to_models) / "catboost_y3.cbm")

    if (cc in ["bbr2", "bbrfrcst"]):
        cc_num = 1
    elif (cc == "cubic"):
        cc_num = 2
    elif (cc == "reno"):
        cc_num = 3

    col = ["Congestion Controller", "Channel RTT (ms)", "Channel Loss (%)", "Channel BW (Kbit/s)"]
    X = pd.DataFrame([[cc_num, rtt, loss, bw],], columns=col)

    predictions = dict()
    for key, model in models.items():
        predictions[key] = model.predict(X)
    '''Emperical alpha/beta below:'''
    predictions[1] = predictions[11] * (0.85) + (0.0) + predictions[13] * (0.15)
    predictions[2] = predictions[21] * (0.25) + predictions[22] * (0.5) + predictions[23] * (0.25)
    predictions[3] = predictions[31] * (0.5) + predictions[32] * (0.3) + predictions[33] * (0.2)
    y_pred = np.array(predictions[1] * (0.85) + predictions[2] * (0.1) + predictions[3] * (0.05))

    print(f"Predicted speed: {y_pred}. Channel parameters:\n\n{X}")

    return y_pred[0] * 0.99636

arg_parser = ArgumentParser(prog='choose_channel',
                            description='')   
arg_parser.add_argument('path_models', metavar='PATH_MODELS', nargs='?', type=str, default='speed_models/',
                        help='Path where the models used are. ')

arg_parser.add_argument('cc', metavar='CC', type=str,
                        help='Congestion controller (reno/cubic/bbr2/bbrfrcst) for dataset. ')
arg_parser.add_argument('rtt', metavar='RTT', type=int,
                        help='Rtt (ms) for dataset. ')
arg_parser.add_argument('loss', metavar='LOSS', type=float,
                        help='Loss (%) for dataset. ')
arg_parser.add_argument('bw', metavar='BW', type=int,
                        help='Bw (Kbit/s) for dataset. ')

if __name__ == "__main__":
    args = arg_parser.parse_args()
    result_vector = load_and_predict(args.path_models, args.cc, args.rtt, args.loss, args.bw)