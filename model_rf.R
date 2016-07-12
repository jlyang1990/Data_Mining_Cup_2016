#' build model (1st layer) using random forest on training dataset and predict on testing dataset
#' trained limited times since this model is time and memory consuming
#' written by Chunzhe Zhang
#' modified by Jilei Yang

rm(list = ls())
library(e1071)
library(randomForest)
library(caret)
library(doParallel)
library(foreach)

#' consider two feature transformation types for new feature items in testing dataset: "new" for entire transformation, "old" for transformation when confident
feature_type <- "new"

#' load feature matrix and response
load(sprintf("feature_data_%s.RData", feature_type))

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

rm(df_all)

#' block cross-validation in time-related prediction problem
#' split 21 months in training dataset into 7 folds
n_fold <- 7
fold_size <- trunc(n_train / n_fold)
fold_id <- c(rep(1:(n_fold - 1), each = fold_size), rep(n_fold, n_train - fold_size * (n_fold-1)))

#' train rf
#' set rf parameters (in caret)
param <- expand.grid(mtry = round(sqrt(ncol(X))))
myControl <- trainControl(method = "none", allowParallel = TRUE)

#' add prediction of testing dataset from each cv fold
y_pred_sum <- rep(0, n_test)
#' mae of each cv fold
scores <- rep(NaN, n_fold)
#' store out of fold prediction of training dataset and predicion of testing dataset from cv
#' used as features in model stacking (2nd layer model)
y_pred_prob_feat <- rep(NaN, n_tot)

cluster <- makeCluster(6, "FORK")
registerDoParallel(cluster)

set.seed(0)

for (i in 1:n_fold) {
  if (i > 1) {
    cat('\n')
  }
  cat(paste('Fold', i, '\n'))
  X_train <- X[fold_id != i, ]
  y_train <- y[fold_id != i]
  X_val <- X[fold_id == i, ]
  y_val <- y[fold_id == i]
  y_index_val <- y_index_cv[fold_id == i]
  
  bst <- train(X_train, as.factor(y_train), method = "parRF", trControl = myControl, tuneGrid = param)
  
  y_pred_prob_feat[1:n_train][fold_id == i] <- predict(bst, X_val, type = "prob")$'1'
  scores[i] <- mean(abs(as.numeric(round(tapply(y_pred_prob_feat[1:n_train][fold_id == i], y_index_val, sum))) - tapply(y_val, y_index_val, sum)))
  cat(paste('\n', 'mae =', scores[i], '\n'))
  y_pred <- predict(bst, X_test, type = "prob")$'1'
  y_pred_sum <- y_pred_sum + y_pred
}

closeAllConnections()

y_pred_prob_feat[(n_train + 1):n_tot] <- y_pred_sum / n_fold
y_pred_prob <- tapply(y_pred_sum / n_fold, y_index_test, sum)
y_pred <- round(y_pred_prob)

cat(paste("mean_score =", mean(scores), "sd_score =", sd(scores), '\n'))

#' save results from 1st layer model for model stacking
file_name <- paste("rf_result", feature_type, sep = "_")
save(y, y_index, scores, y_pred, y_pred_prob, y_pred_prob_feat, ind_drop, file = paste(file_name, ".RData", sep = ""))