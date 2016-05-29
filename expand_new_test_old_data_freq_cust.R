rm(list=ls())
library(xgboost)

load("expand_data_old.RData")

# expand df_all
y_index = rep(1:length(df_all$quantity), times=df_all$quantity)
df_all = df_all[rep(1:nrow(df_all), times=df_all$quantity), ]
y = rep(rep(c(1,0), length(y)), times = c(t(cbind(y, df_train_quantity-y))))
n_train = length(y)
n_tot = length(y_index)
n_test = n_tot-n_train

print(dim(df_all))

#df_all$articleID = df_all$articleID-1e6
#df_all$customerID = df_all$customerID-1e6
# need to make sure that all elements in df_all are numeric (double or integer)
X_all = as.matrix(df_all)
# num of order positions per customer
freq_per_customer = df_all$freq_per_customer
rm(df_all)
X = X_all[1:n_train, ]
X_test = X_all[(n_train+1):n_tot, ]
rm(X_all)

# likelihood generator ########################################################################################################
likelihood_train_generator = function(X_train, y, object, expand=F, random=F){
  if(expand){
    likelihood_train = (tapply(y, as.character(X_train[, object]), sum)[as.character(X_train[, object])]-y)/(tapply(y, as.character(X_train[, object]), length)[as.character(X_train[, object])]-1)
    likelihood_train[is.na(likelihood_train)] = ((sum(y)-y)/(nrow(X_train)-1))[is.na(likelihood_train)]
  }else{
    likelihood_train = (tapply(y, as.character(X_train[, object]), sum)[as.character(X_train[, object])]-y)/(tapply(X_train[, "quantity"], as.character(X_train[, object]), sum)[as.character(X_train[, object])]-X_train[, "quantity"])
    likelihood_train[is.na(likelihood_train)] = ((sum(y)-y)/(sum(X_train[, "quantity"])-X_train[, "quantity"]))[is.na(likelihood_train)]
  }
  if(random){
    likelihood_train = pmin(likelihood_train*rnorm(nrow(X_train), 1, 0.02), 1)
  }
  return(likelihood_train)
}

likelihood_test_generator = function(X_train, y, X_test, object, expand=F){
  new_object = setdiff(as.character(X_test[, object]), as.character(X_train[, object]))
  if(expand){
    temp = rep(sum(y)/nrow(X_train), length(new_object))
    names(temp) = new_object
    likelihood_test = c(tapply(y, as.character(X_train[, object]), sum)/tapply(y, as.character(X_train[, object]), length), temp)[as.character(X_test[, object])]
  }else{
    temp = rep(sum(y)/sum(X_train[, "quantity"]), length(new_object))
    names(temp) = new_object
    likelihood_test = c(tapply(y, as.character(X_train[, object]), sum)/tapply(X_train[, "quantity"], as.character(X_train[, object]), sum), temp)[as.character(X_test[, object])]
  }
  return(likelihood_test)
}

#likelihood_list = c('customerID', 'articleID', 'colorCode')
likelihood_list = c('customerID')

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
param = list(eta = 0.1, 
             subsample = 0.95, 
             colsample_bytree = 0.8,
             max_depth = 6,
             silent = 1, 
             objective = 'binary:logistic', 
             eval_metric = 'logloss')

scores = rep(NaN, n_fold)

y_index_test = y_index[(n_train+1):n_tot]
y_index_cv = y_index[1:n_train]
freq_per_customer_cv = freq_per_customer[1:n_train]
y_pred_sum = rep(0, n_test)

y_pred_prob_feat = rep(NaN, n_tot)

set.seed(0)
for (i in 1:n_fold){
  if (i>1){
    cat('\n')
  }
  cat(paste('Fold', i, '\n'))
  X_train = X[fold_id!=i & freq_per_customer_cv>1, ]
  X_val = X[fold_id==i, ]
  y_train = y[fold_id!=i & freq_per_customer_cv>1]
  y_val = y[fold_id==i]
  y_index_val = y_index_cv[fold_id==i]
  #for(l in likelihood_list){X_train = cbind(X_train, likelihood_train_generator(X_train, y_train, l, expand=T, random=T))}
  #for(l in likelihood_list){X_val = cbind(X_val, likelihood_test_generator(X_train, y_train, X_val, l, expand=T))}
  dtrain = xgb.DMatrix(X_train, label=y_train)
  dval = xgb.DMatrix(X_val, label=y_val)
  watchlist = list(eval=dval, train=dtrain)
  bst = xgb.train(param, dtrain, nthread=40, nrounds=1e4, watchlist, early.stop.round=100, maximize=FALSE)
  y_pred_prob_feat[1:n_train][fold_id==i] = predict(bst, X_val, ntreelimit=bst$bestInd)
  scores[i] = mean(abs(as.numeric(round(tapply(y_pred_prob_feat[1:n_train][fold_id==i], y_index_val, sum)))-tapply(y_val, y_index_val, sum)))
  cat(paste('\n', 'mae =', scores[i], '\n'))
  y_pred = predict(bst, X_test, ntreelimit=bst$bestInd)
  y_pred_sum = y_pred_sum+y_pred
}

y_pred_prob_feat[(n_train+1):n_tot] = y_pred_sum/n_fold
y_pred_prob = tapply(y_pred_sum/n_fold, y_index_test, sum)
y_pred = round(y_pred_prob)

cat(paste(mean(scores), sd(scores), '\n'))

save(y, y_index, scores, y_pred, y_pred_prob, y_pred_prob_feat, ind_drop, file = "DMC16-expand-test-old-data-freq-cust.RData")