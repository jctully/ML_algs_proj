import numpy as np 
import pandas as pd 
from scipy.io import loadmat
import xgboost as xgb
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDRegressor, LinearRegression, RidgeCV
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import argparse
import sys
import pickle
import wandb


def main(argv):
    args = parse_all_args()

    #setup wandb
    print("Syncing with wandb!")
    hyperparameter_defaults = dict(
        penalty     = args.penalty,
        lr_type     = args.lr_type,
        lr          = args.lr,
        alpha       = args.alpha   
    )
    wandb.init(project='ml_proj', config=hyperparameter_defaults)
    config = wandb.config
    experiment_name = config.penalty + "_lr_type_" + config.lr_type + "_" + str(config.lr) + "_alpha" + str(config.alpha)
    print(experiment_name)
    wandb.run.name = experiment_name
    wandb.run.save()

    DATA_DIR = '/home/harri267/file_share/task_1'

    # task 1
    # load orig data
    data = loadmat(f"{DATA_DIR}/data.mat")
    # X_train = data['X_train']
    # n,d = X_train.shape
    # #print(X_train.shape)
    # X_test = data['X_test']
    y_train = data['y_train'].squeeze()
    # print("loaded data...")

    #load data
    with open("X_train.pkl", "rb" ) as f:
        X_train = pickle.load(f)
    with open("X_test.pkl", "rb" ) as f:
        X_test = pickle.load(f)
    print("loaded data...")

    #split data into dev set
    X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.1, random_state=0)
    print("split data...")

    #scale input data
    # sc = StandardScaler()
    # X_train = sc.fit_transform(X_train)
    # # X_dev = sc.transform(X_dev)
    # print("scaled data...")

    #transform data with PCA
    # pca = PCA(n_components=2000)
    # X_train = pca.fit_transform(X_train)
    # X_test = pca.transform(X_test)
    # with open("X_train.pkl", "wb" ) as f:
    #     pickle.dump(X_train, f )
    # with open("X_test.pkl", "wb" ) as f:
    #     pickle.dump(X_test, f )
    # print("transformed data with PCA...")
    # exit()

    #perform sklearn regression with SGD
    #sgdr = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1])
    sgdr = SGDRegressor(penalty = args.penalty, alpha=args.alpha, learning_rate=args.lr_type, eta0=args.lr)
    print("training...")
    sgdr.fit(X_train, y_train)
    # print(sgdr.n_iter_)
    print(f"trained for {sgdr.n_iter_} iter")
    train_preds = sgdr.predict(X_train)
    dev_preds = sgdr.predict(X_dev)

    train_err = sklearn.metrics.median_absolute_error(y_train, train_preds)
    dev_err = sklearn.metrics.median_absolute_error(y_dev, dev_preds)
    print("train error:", train_err)
    print("dev error:", dev_err)

    stats = {
        "n_iter": sgdr.n_iter_,
        "train_med_ae": train_err,
        "dev_med_ae": dev_err
    }
    wandb.log(stats)

def parse_all_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--penalty",
                        type=str,
                        help="The penalty type (str) [default: 'l2']",
                        default='l2')
    parser.add_argument("--lr_type",
                        type=str,
                        help="The learning rate type (str) [default: 'optimal']",
                        default='optimal')
    parser.add_argument("--lr",
                        type=float,
                        help="The learning rate (float) [default: 0.01]",
                        default=0.01)
    parser.add_argument("--alpha",
                        type=float,
                        help="The alpha term (float) [default: 0.0001]",
                        default=0.0001)
    return parser.parse_args()

if __name__ == "__main__":
    main(sys.argv)