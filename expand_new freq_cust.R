rm(list=ls())
R_dir = "/home/minjay/R/x86_64-pc-linux-gnu-library/3.2"
library(xgboost, lib.loc=R_dir)

load("expand_data_new.RData")

# new changes to df_all

# drop orderID since
# (i) its high cardinality;
# (ii) the information it contains can be mostly explained by order date and customer ID.
df_all = subset(df_all, select=-orderID)

# drop voucherID since voucherAmount is more important
df_all = subset(df_all, select=-voucherID)

# drop price
df_all = subset(df_all, select=-price)

# drop orderDate
df_all = subset(df_all, select=-orderDate)

# drop first orderDate
df_all = subset(df_all, select=-first_orderDate)

# drop rrp_new
df_all = subset(df_all, select=-rrp_new)

# drop item_index
df_all = subset(df_all, select=-c(choice_item_index, ac_item_index, as_item_index, cp_item_index, sp_item_index, csp_item_index, pr_item_index, cpr_item_index, spr_item_index, cspr_item_index, choice_1_item_index, as_1_item_index, sp_1_item_index, pr_1_item_index, spr_1_item_index, choice_234_item_index, as_234_item_index, sp_234_item_index, pr_234_item_index, spr_234_item_index, choice_34_item_index, as_34_item_index, sp_34_item_index, pr_34_item_index, spr_34_item_index, choice_4_item_index, as_4_item_index, sp_4_item_index, pr_4_item_index, spr_4_item_index))

# drop temp
df_all = subset(df_all, select=-c(temp_ac, temp_as, temp_cp, temp_sp, temp_csp, temp_pr, temp_cpr, temp_spr, temp_cspr, temp_choice_1, temp_as_1, temp_sp_1, temp_pr_1, temp_spr_1, temp_choice_234, temp_as_234, temp_sp_234, temp_pr_234, temp_spr_234, temp_choice_34, temp_as_34, temp_sp_34, temp_pr_34, temp_spr_34, temp_choice_4, temp_as_4, temp_sp_4, temp_pr_4, temp_spr_4))

OHE_feats = c('order_weekday', 'sizeCode', 'productGroup', 'deviceID', 'paymentMethod')
df_all = df_all[, !(names(df_all) %in% OHE_feats)]

# expand df_all
df_all = df_all[rep(1:nrow(df_all), times=df_all$quantity), ]
y = rep(rep(c(1,0), length(y)), times = c(t(cbind(y, df_train_quantity-y))))
n_train = length(y)
y_index = rep(1:length(df_train_quantity), times=df_train_quantity)

print(dim(df_all))

#df_all$articleID = df_all$articleID-1e6
#df_all$customerID = df_all$customerID-1e6
# need to make sure that all elements in df_all are numeric (double or integer)
X_all = as.matrix(df_all)
freq_per_customer = df_all$freq_per_customer[1:n_train]
rm(df_all)
X = X_all[1:n_train, ]
X_test = X_all[(n_train+1):nrow(X_all), ]
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
             subsample = 0.5, 
             colsample_bytree = 0.9,
             max_depth = 6,
             silent = 1, 
             objective = 'binary:logistic', 
             eval_metric = 'logloss')

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
freq_per_customer_cv = freq_per_customer[fold_id!=n_fold]
y_pred_sum = rep(0, length(y_hold_out))
rm(X)

y_pred_prob_feat = rep(NaN, n_train)

set.seed(0)
for (i in 1:(n_fold-1)){
  if (i>1){
    cat('\n')
  }
  cat(paste('Fold', i, '\n'))
  X_train = X_cv[fold_id_cv!=i & freq_per_customer_cv>1, ]
  X_val = X_cv[fold_id_cv==i, ]
  y_train = y_cv[fold_id_cv!=i & freq_per_customer_cv>1]
  y_val = y_cv[fold_id_cv==i]
  y_index_val = y_index_cv[fold_id_cv==i]
  #for(l in likelihood_list){X_train = cbind(X_train, likelihood_train_generator(X_train, y_train, l, expand=T, random=T))}
  #for(l in likelihood_list){X_val = cbind(X_val, likelihood_test_generator(X_train, y_train, X_val, l, expand=T))}
  dtrain = xgb.DMatrix(X_train, label=y_train)
  dval = xgb.DMatrix(X_val, label=y_val)
  watchlist = list(eval=dval, train=dtrain)
  bst = xgb.train(param, dtrain, nthread=20, nrounds=1e4, watchlist, early.stop.round=100, maximize=FALSE)
  y_pred_prob_feat[fold_id==i] = predict(bst, X_val, ntreelimit=bst$bestInd)
  scores[i] = mean(abs(as.numeric(round(tapply(y_pred_prob_feat[fold_id==i], y_index_val, sum)))-tapply(y_val, y_index_val, sum)))
  cat(paste('\n', 'mae =', scores[i], '\n'))
  y_pred = predict(bst, X_hold_out, ntreelimit=bst$bestInd)
  y_pred_sum = y_pred_sum+y_pred
}

y_pred_prob_feat[fold_id==n_fold] = y_pred_sum/(n_fold-1)
y_pred_prob = tapply(y_pred_prob_feat[fold_id==n_fold], y_index_hold_out, sum)
y_pred = round(y_pred_prob)
y_hold_out = tapply(y_hold_out, y_index_hold_out, sum)

cat(paste(mean(scores), sd(scores), mean(abs(y_pred-y_hold_out)), '\n'))

save(y, y_index, y_hold_out, scores, y_pred, y_pred_prob, y_pred_prob_feat, file = "DMC16-expand-new-freq-cust.RData")