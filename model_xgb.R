#' build model (1st layer) using xgboost on training dataset and predict on testing dataset
#' written by Minjie Fan
#' modified by Jilei Yang

rm(list = ls())
library(xgboost)

#' consider two feature transformation types for new feature items in testing dataset: "new" for entire transformation, "old" for transformation when confident
feature_type <- "new"

#' drop features of total price per customer per day/3days/5days/7days/11days/15days if drop_feat <- TRUE
drop_feat <- FALSE

#' drop articles with price equal 0 (regarded as gift) in training and testing dataset, and set the predictions of gifts in testing dataset equal 0 if drop_gift <- TRUE
drop_gift <- FALSE

#' drop customers with only one record (one row) in training dataset if drop_low_freq_cust <- TRUE
drop_low_freq_cust <- FALSE

#' add (out-of-sample) likelihood feature with respect to customer if add_likelihood_cust <- TRUE
add_likelihood_cust <- FALSE

#' add (out-of-sample) likelihood feature with respect to month if add_likelihood_month <- TRUE
add_likelihood_month <- FALSE

#' set the value for parameter "subsample" in xgboost
xgb_subsample <- 0.95

#' set the value for parameter "colsample_bytree" in xgboost
xgb_colsample <- 0.8

#' load feature matrix and response
load(sprintf("feature_data_%s.RData", feature_type))

if(drop_feat == TRUE) {
  df_all <- subset(df_all, select = -c(price_per_cust_date, price_per_cust_3date, price_per_cust_5date, price_per_cust_7date, price_per_cust_11date, price_per_cust_15date))
}

#' expand feature matrix "df_all" and response "y" with respect to "quantity"
#' new response "y" is binary: greatly reduce computational complexity
y_index <- rep(1:length(df_all$quantity), times = df_all$quantity)
df_all <- df_all[rep(1:nrow(df_all), times = df_all$quantity), ]
y <- rep(rep(c(1, 0), length(y)), times = c(t(cbind(y, df_train_quantity - y))))
n_train <- length(y)
n_tot <- length(y_index)
n_test <- n_tot - n_train
print(dim(df_all))

#' construct feature matrices for training and testing
df_all <- as.matrix(df_all)
X <- df_all[1:n_train, ]
X_test <- df_all[(n_train + 1):n_tot, ]
y_index_cv <- y_index[1:n_train]
y_index_test <- y_index[(n_train + 1):n_tot]

if(drop_gift == TRUE) {
  price_per_quantity <- df_all$price_per_quantity
  price_per_quantity_cv <- price_per_quantity[1:n_train]
  price_per_quantity_test <- price_per_quantity[(n_train + 1):n_tot]
}
if(drop_low_freq_cust == TRUE) {
  freq_per_customer <- df_all$freq_per_customer
  freq_per_customer_cv <- freq_per_customer[1:n_train]
}
rm(df_all)

#' likelihood generator function for training dataset
LikelihoodTrainGenerator <- function(X_train, y, object, noise_sd = 0.02) {
  likelihood_train <- (tapply(y, as.character(X_train[, object]), sum)[as.character(X_train[, object])] - y) / (tapply(X_train[, "quantity"], as.character(X_train[, object]), sum)[as.character(X_train[, object])] - X_train[, "quantity"])
  likelihood_train[is.na(likelihood_train)] <- ((sum(y) - y) / (sum(X_train[, "quantity"]) - X_train[, "quantity"]))[is.na(likelihood_train)]
  likelihood_train <- pmin(likelihood_train * rnorm(nrow(X_train), 1, noise_sd), 1)
  return(likelihood_train)
}

#' likelihood generator function for testing dataset
LikelihoodTestGenerator <- function(X_train, y, X_test, object){
  new_object <- setdiff(as.character(X_test[, object]), as.character(X_train[, object]))
  temp <- rep(sum(y) / nrow(X_train), length(new_object))
  names(temp) <- new_object
  likelihood_test <- c(tapply(y, as.character(X_train[, object]), sum) / tapply(y, as.character(X_train[, object]), length), temp)[as.character(X_test[, object])]
  return(likelihood_test)
}

#' construct likelihood list
likelihood_list <- c()
if(add_likelihood_cust == TRUE) {
  likelihood_list <- c("customerID")
}
if(add_likelihood_month == TRUE) {
  likelihood_list <- c("order_month")
}

#' add likelihood feature to testing dataset
for(likelihood_term in likelihood_list) {
  X_test <- cbind(X_test, LikelihoodTestGenerator(X, y, X_test, likelihood_term))
}

#' block cross-validation in time-related prediction problem
#' split 21 months in training dataset into 7 folds
n_fold <- 7
fold_size <- trunc(n_train / n_fold)
fold_id <- c(rep(1:(n_fold - 1), each = fold_size), rep(n_fold, n_train - fold_size * (n_fold-1)))

#' train xgboost
#' set xgboost parameters
param <- list(eta = 0.1, 
              subsample = xgb_subsample, 
              colsample_bytree = xgb_colsample,
              max_depth = 6,
              silent = 1, 
              objective = 'binary:logistic', 
              eval_metric = 'logloss')

#' add prediction of testing dataset from each cv fold
y_pred_sum <- rep(0, n_test)
#' mae of each cv fold
scores <- rep(NaN, n_fold)
#' store out of fold prediction of training dataset and predicion of testing dataset from cv
#' used as features in model stacking (2nd layer model)
y_pred_prob_feat <- rep(NaN, n_tot)

set.seed(0)

for (i in 1:n_fold) {
  if (i > 1) {
    cat('\n')
  }
  cat(paste('Fold', i, '\n'))
  if(drop_gift == TRUE) {
    X_train <- X[fold_id != i & price_per_quantity_cv != 0, ]
    y_train <- y[fold_id != i & price_per_quantity_cv != 0]
  } else if(drop_low_freq_cust == TRUE) {
    X_train <- X[fold_id != i & freq_per_customer_cv > 1, ]
    y_train <- y[fold_id != i & freq_per_customer_cv > 1]
  } else {
    X_train <- X[fold_id != i, ]
    y_train <- y[fold_id != i]
  }
  X_val <- X[fold_id == i, ]
  y_val <- y[fold_id == i]
  y_index_val <- y_index_cv[fold_id == i]
  for(likelihood_term in likelihood_list){
    X_train <- cbind(X_train, LikelihoodTrainGenerator(X_train, y_train, likelihood_term))
    X_val <- cbind(X_val, LikelihoodTestGenerator(X_train, y_train, X_val, likelihood_term))
  }
  
  dtrain <- xgb.DMatrix(X_train, label = y_train)
  dval <- xgb.DMatrix(X_val, label = y_val)
  watchlist <- list(eval = dval, train = dtrain)
  bst <- xgb.train(param, dtrain, nthread = 40, nrounds = 1e4, watchlist, early.stop.round = 100, maximize = FALSE)
  
  y_pred_prob_feat[1:n_train][fold_id == i] <- predict(bst, X_val, ntreelimit = bst$bestInd)
  if(drop_gift == TRUE) {
    y_pred_prob_feat[1:n_train][fold_id == i & price_per_quantity_cv == 0] <- 0
  }
  scores[i] <- mean(abs(as.numeric(round(tapply(y_pred_prob_feat[1:n_train][fold_id == i], y_index_val, sum)))-tapply(y_val, y_index_val, sum)))
  cat(paste('\n', 'mae =', scores[i], '\n'))
  y_pred <- predict(bst, X_test, ntreelimit = bst$bestInd)
  y_pred_sum <- y_pred_sum + y_pred
}

y_pred_prob_feat[(n_train + 1):n_tot] <- y_pred_sum / n_fold
if(drop_gift == TRUE) {
  y_pred_prob_feat[(n_train + 1):n_tot & price_per_quantity_test == 0] <- 0
}
y_pred_prob <- tapply(y_pred_sum / n_fold, y_index_test, sum)
y_pred <- round(y_pred_prob)

cat(paste("mean_score =", mean(scores), "sd_score =", sd(scores), '\n'))

#' save results from 1st layer model for model stacking
save_list <- c(drop_feat, drop_gift, drop_low_freq_cust, add_likelihood_cust, add_likelihood_month)
names(save_list) <- c("drop_feat", "drop_gift", "drop_low_freq_cust", "add_likelihood_cust", "add_likelihood_month")
file_name <- paste("xgb_result", feature_type, ifelse(length(l[l == T]) == 0, "base", names(l[l == T])), xgb_subsample * 100, xgb_colsample * 100, sep = "_")
save(y, y_index, scores, y_pred, y_pred_prob, y_pred_prob_feat, ind_drop, file = paste(file_name, ".RData", sep = ""))