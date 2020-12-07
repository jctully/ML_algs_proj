import numpy as np 
import pandas as pd 
from scipy.io import loadmat
import xgboost as xgb
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

DATA_DIR = '/home/harri267/file_share/task_1'

# task 1
#load data
data = loadmat(f"{DATA_DIR}/data.mat")
X_train = data['X_train']
n,d = X_train.shape
#print(X_train.shape)
X_test = data['X_test']
y_train = data['y_train'].squeeze()
print("loaded data...")

#split data into dev set
X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
print("split data...")

#scale input data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_dev = sc.transform(X_dev)
print("scaled data...")

# perform PCA experiments
component_list = np.logspace(9, 11, base=2, num=10)
component_list = [int(i) for i in component_list if i < d]
pct_exp = []
for ncomp in component_list:
    pca = PCA(n_components=ncomp)
    pca.fit(X_train)
    # X_train = pca.fit_transform(X_train)
    # X_dev = pca.transform(X_dev)
    explained_variance = pca.explained_variance_ratio_
    pct_exp.append(np.sum(explained_variance))
    print(ncomp, np.sum(explained_variance))

plt.plot(component_list, pct_exp)
plt.plot(component_list, [0.95 for _ in component_list])
plt.xlabel("# of Components")
plt.ylabel("% variance explained")
plt.xscale('log')
plt.axis([500,2048,0,1])
plt.grid(True)
plt.savefig("fig")
plt.title("Components vs. variance explained")