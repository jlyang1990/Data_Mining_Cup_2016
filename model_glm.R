# second model 
# Data Mining Cup 2016
# written by Minjie Fan, 2016
# modified by Jilei Yang, 2016

rm(list=ls())
library(stringr)
library(plyr)


### Minjie, please load your data frame
# including df_all, y, y_index with the same names
# otherwise you can run the 260 feature to the X_all=as.matrix(df_all)
load(file = "baseFeature_oldSC_noLik.RData")

n_train = length(y)
# need to make sure that all elements in df_all are numeric (double or integer)
X_all = as.matrix(df_all)
X = X_all[1:n_train, ]
X_test = X_all[(n_train+1):nrow(X_all), ]
#for(l in likelihood_list){X_test = cbind(X_test, likelihood_test_generator(X, y, X_test, l, expand=T))}

my_mae = function(preds, dtrain){
  labels = getinfo(dtrain, 'label')
  pred_labels = as.numeric(preds>0.5)
  mae = mean(abs(labels-pred_labels))
  return(list(metric = 'mae', value = mae))
}

# block cross-validation
# take the idea from block bootstrap
# 21 months in training data
n_fold = 7
fold_size = trunc(n_train/n_fold)
fold_id = c(rep(1:(n_fold-1), each=fold_size), rep(n_fold, n_train-fold_size*(n_fold-1)))

# train xgboost
# softmax gives class instead of prob
#param = list(eta = 0.1, 
#             subsample = 0.5, 
#             colsample_bytree = 0.9,
#             max_depth = 6,
#             silent = 1, 
#             objective = 'binary:logistic', 
#             eval_metric = my_mae)

scores = rep(NaN, (n_fold-1))
scores_exp = rep(NaN, (n_fold-1))

X_hold_out = X[fold_id==n_fold, ]
y_hold_out = y[fold_id==n_fold]
y_index_hold_out = y_index[fold_id==n_fold]
X_cv = X[fold_id<n_fold, ]
y_cv = y[fold_id<n_fold]
y_index_cv = y_index[fold_id<n_fold]
#for(l in likelihood_list){X_hold_out = cbind(X_hold_out, likelihood_test_generator(X_cv, y_cv, X_hold_out, l, expand=T))}
fold_id_cv = fold_id[fold_id!=n_fold]
y_pred_sum = rep(0, length(y_hold_out))

set.seed(0)
for (i in 1:(n_fold-1)){
  if (i>1){
    cat('\n')
  }
  cat(paste('Fold', i, '\n'))
  X_train = X_cv[fold_id_cv!=i, ]
  X_val = X_cv[fold_id_cv==i, ]
  y_train = y_cv[fold_id_cv!=i]
  y_val = y_cv[fold_id_cv==i]
  y_index_val = y_index_cv[fold_id_cv==i]
  #for(l in likelihood_list){X_train = cbind(X_train, likelihood_train_generator(X_train, y_train, l, expand=T))}
  #for(l in likelihood_list){X_val = cbind(X_val, likelihood_test_generator(X_train, y_train, X_val, l, expand=T))}
  # fit GLM for each fold
  dataGLMfold = data.frame(y_train, X_train)
  glmfitfold = glm(y_train~., data = dataGLMfold, family = binomial())
  pred_val_exp_label = predict(glmfitfold, newdata=data.frame(X_val), type="response") # fitted prob.
  scores_exp[i] = mean(abs(y_val-pred_val_exp_label))
  scores[i] = mean(abs(as.numeric(round(tapply(pred_val_exp_label, y_index_val, sum)))-
                         tapply(y_val, y_index_val, sum)))
  cat(paste('\n', 'mae =', scores[i], '\n'))
  y_pred = predict(glmfitfold, newdata=data.frame(X_hold_out), type="response") # fitted prob.
  y_pred_sum = y_pred_sum+y_pred
  
}

y_pred = round(tapply(y_pred_sum/(n_fold-1), y_index_hold_out, sum))
y_hold_out_exp = y_hold_out
y_hold_out = tapply(y_hold_out, y_index_hold_out, sum)

cat(paste(mean(scores), sd(scores), mean(abs(y_pred-y_hold_out)), '\n'))

save(y_hold_out, y_hold_out_exp, scores, scores_exp, y_pred, y_pred_sum, file = "DMC16-GLMbase_Exp_oldSC_noLik.RData")