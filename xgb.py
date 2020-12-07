


dtrain = xgb.DMatrix(X_train, label=y_train)
ddev = xgb.DMatrix(X_dev, label=y_dev)

# Set model params
xgb_params = {
    'objective'         : 'reg:squarederror',
    'eval_metric'       : 'mae',
    'eta'               : 0.3,
    'nthread'           : 8,
}
evallist = [(dtrain, 'train'), (ddev, 'dev')]
num_round = 10
print("Training...")
bst = xgb.train(xgb_params, dtrain, num_round, evallist, early_stopping_rounds=500)
ypred = bst.predict(dtrain)
err = sklearn.metrics.median_absolute_error(y_train, ypred)
print(err)

#task 2
# DATA_DIR = '/home/harri267/file_share/task_2'
# data = loadmat(f"{DATA_DIR}/data.mat")
# X_train = data['X_train']
# X_test = data['X_test']
# y_train = data['y_train'].squeeze()
# print(y_train.shape)

# dtrain = xgb.DMatrix(X_train, label=y_train)
# # Set model params
# xgb_params = {
#         'objective'         : 'multi:softmax',
#         'eval_metric'       : 'merror',
#         'num_class'         : 3,
#         'nthread'           : 8,
#         'eta'               : 0.3,
#     }
# evallist = [(dtrain, 'train')]
# num_round = 15
# bst = xgb.train(xgb_params, dtrain, num_round, evallist, early_stopping_rounds=500)
# ypred = bst.predict(dtrain)
# err = sklearn.metrics.balanced_accuracy_score(y_train, ypred)
# print(err)