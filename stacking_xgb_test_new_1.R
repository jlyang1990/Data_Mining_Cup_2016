library(xgboost)

# load pred
load("stacking_new.RData")

# combine
dim(X_all)
n_train = length(y)
n_tot = length(y_index)
n_test = n_tot-n_train

X = X_all[1:n_train, ]
X_test = X_all[(n_train+1):n_tot, ]

# block cross-validation
# take the idea from block bootstrap
# 21 months in training data
n_fold = 7
fold_size = trunc(n_train/n_fold)
fold_id = c(rep(1:(n_fold-1), each=fold_size), rep(n_fold, n_train-fold_size*(n_fold-1)))

# train xgboost
# softmax gives class instead of prob
param = list(eta = 0.01,
             subsample = 0.6, 
             colsample_bytree = 0.9,
             max_depth = 6,
             silent = 1, 
             objective = 'binary:logistic', 
             eval_metric = 'logloss')

scores = rep(NaN, n_fold)

y_index_test = y_index[(n_train+1):n_tot]
y_index_cv = y_index[1:n_train]
y_pred_sum = rep(0, n_test)

set.seed(0)
for (i in 1:3){
  if (i>1){
    cat('\n')
  }
  cat(paste('Fold', i, '\n'))
  X_train = X[fold_id!=i, ]
  X_val = X[fold_id==i, ]
  y_train = y[fold_id!=i]
  y_val = y[fold_id==i]
  y_index_val = y_index_cv[fold_id==i]
  dtrain = xgb.DMatrix(X_train, label=y_train)
  dval = xgb.DMatrix(X_val, label=y_val)
  watchlist = list(eval=dval, train=dtrain)
  bst = xgb.train(param, dtrain, nthread=40, nrounds=1e4, watchlist, early.stop.round=100, maximize=FALSE)
  scores[i] = mean(abs(as.numeric(round(tapply(predict(bst, X_val, ntreelimit=bst$bestInd), y_index_val, sum)))-tapply(y_val, y_index_val, sum)))
  cat(paste('\n', 'mae =', scores[i], '\n'))
  y_pred = predict(bst, X_test, ntreelimit=bst$bestInd)
  y_pred_sum = y_pred_sum+y_pred
}

y_pred_prob = tapply(y_pred_sum/n_fold, y_index_test, sum)
y_pred = round(y_pred_prob)

save(y_pred_prob, y_pred, file="stacking_xgb_test_new_1.RData")