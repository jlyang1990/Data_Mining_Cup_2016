#' build stacking model (2nd layer) using xgboost on training dataset and predict on testing dataset
#' written by Minjie Fan
#' modified by Jilei Yang

rm(list = ls())
library(xgboost)

#' consider two feature transformation types for new feature items in testing dataset: "new" for entire transformation, "old" for transformation when confident
feature_type <- "new"

#' load feature matrix and response
load(sprintf("stacking_data_%s.RData", feature_type))

#' construct feature matrices for training and testing
n_train <- length(y)
n_tot <- length(y_index)
n_test <- n_tot - n_train
X <- X_all[1:n_train, ]
X_test <- X_all[(n_train + 1):n_tot, ]
y_index_cv <- y_index[1:n_train]
y_index_test <- y_index[(n_train + 1):n_tot]

#' block cross-validation in time-related prediction problem
#' split 21 months in training dataset into 7 folds
n_fold <- 7
fold_size <- trunc(n_train / n_fold)
fold_id <- c(rep(1:(n_fold - 1), each = fold_size), rep(n_fold, n_train - fold_size * (n_fold - 1)))

#' train xgboost
#' set xgboost parameters
param <- list(eta = 0.01,
              subsample = 0.6, 
              colsample_bytree = 0.9,
              max_depth = 6,
              silent = 1, 
              objective = 'binary:logistic', 
              eval_metric = 'logloss')

#' add prediction of testing dataset from each cv fold
y_pred_sum = rep(0, n_test)
#' mae of each cv fold
scores = rep(NaN, n_fold)

set.seed(0)

for (i in 1:n_fold)n{
  if (i > 1) {
    cat('\n')
  }
  cat(paste('Fold', i, '\n'))
  X_train <- X[fold_id != i, ]
  y_train <- y[fold_id != i]
  X_val <- X[fold_id == i, ]
  y_val <- y[fold_id == i]
  y_index_val <- y_index_cv[fold_id == i]
  
  dtrain <- xgb.DMatrix(X_train, label = y_train)
  dval <- xgb.DMatrix(X_val, label = y_val)
  watchlist <- list(eval = dval, train = dtrain)
  bst <- xgb.train(param, dtrain, nthread = 40, nrounds = 1e4, watchlist, early.stop.round = 100, maximize = FALSE)
  
  scores[i] <- mean(abs(as.numeric(round(tapply(predict(bst, X_val, ntreelimit = bst$bestInd), y_index_val, sum))) - tapply(y_val, y_index_val, sum)))
  cat(paste('\n', 'mae =', scores[i], '\n'))
  y_pred <- predict(bst, X_test, ntreelimit = bst$bestInd)
  y_pred_sum <- y_pred_sum + y_pred
}

y_pred_prob <- tapply(y_pred_sum / n_fold, y_index_test, sum)
y_pred <- round(y_pred_prob)

#' save results from 2nd layer model
save(y_pred_prob, y_pred, ind_drop, file = sprintf("stacking_result_%s.RData", feature_type))