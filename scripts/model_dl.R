#' build model (1st layer) using deep learning on training dataset and predict on testing dataset
#' written by Hao Ji
#' modified by Jilei Yang

rm(list = ls())
library(h2o)

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

#' tell DL that it is classification problem
y <- factor(y)

#' select subset of features for DL execution for speedup
load(sprintf("xgb_result_%s_base_95_80.RData", feature_type))
n_feat <- 300
imp_var <- importance_matrix[1:n_feat, 1]
df_all <- df_all[, (names(df_all) %in% imp_var)]
print(paste("Most important", ncol(df_all), "features are used.\n", sep=" "))

#' block cross-validation in time-related prediction problem
#' split 21 months in training dataset into 7 folds
n_fold <- 7
fold_size <- trunc(n_train / n_fold)
fold_id <- c(rep(1:(n_fold - 1), each = fold_size), rep(n_fold, n_train - fold_size * (n_fold-1)))

#' Randomly specifies the parameters for DL
#' Make sure that we do not fix any random seed for different execution
hidden_list <- list(c(200, 150, 100), c(120, 90, 65, 40), c(300, 200), 
                   c(40, 40, 40, 40, 40), c(100, 100, 50, 50), c(150, 150, 150))
hidden_layer <- hidden_list[[sample(1:length(hidden_list), 1)]]
hidden_dropout_ratio <- c(sample(c(0.1, 0.25, 0.4), 1), rep(0, length(hidden_layer) - 1))
l1_penalty <- sample(c(1e-5, 1e-4, 1e-3), 1)
l2_penalty <- sample(c(1e-5, 1e-4, 1e-3), 1)

#' store out of fold prediction of training dataset and predicion of testing dataset from cv
#' used as features in model stacking (2nd layer model)
y_pred_prob_feat <- rep(NaN, n_tot)

#' train h2o.deeplearning
#' DL node initialization
h2o.init(startH2O = TRUE, nthreads = -1)
h2o.removeAll()

#' link data with h2o.init
train <- as.h2o(data.frame(df_all[1:n_train, ], response = y, foldcol = fold_id))
test <- as.h2o(df_all[(n_train + 1):nrow(df_all), ])

#' names of the predictors and response
predictors <- names(df_all)
response <- "response"

#' execute DL for cross validation, feed data to h2o, as H2OFrame
#' automatic CV and model fitting based on whole data using epochs estimated based on CV
model_dl <- h2o.deeplearning(x = 1:(ncol(train) - 2), 
                             y = response, 
                             training_frame = train,
                             model_id = "model_dl",  # validation set kept empty for automatic CV
                             standardize = TRUE,  # default
                             activation = "RectifierWithDropout",  # changable
                             hidden = hidden_layer,  # changable, architecture
                             epochs = 10000,
                             stopping_rounds = 5,  # changable
                             stopping_metric = "logloss",  # changable
                             stopping_tolerance = 1e-3,  # changable
                             # seed = 2016,
                             variable_importances = TRUE,
                             adaptive_rate = TRUE,
                             # rho = 0.99,
                             # epsilon = 1e-7,
                             input_dropout_ratio = 0,
                             hidden_dropout_ratios = hidden_dropout_ratio,
                             l1 = l1_penalty,  # changable
                             l2 = l2_penalty,  # changable
                             # loss = "Automatic",
                             score_training_samples = 100000,
                             score_validation_samples = 0,
                             score_duty_cycle = 0.1,  # 10% at most for scoring validation data
                             max_runtime_secs = 0,  # default, disabled for now
                             missing_values_handling = "MeanImputation",  # default
                             # fast_mode = FALSE,  # disabled for accuracy
                             # nfolds = 7,  # no need once fold_column given
                             fold_column = "foldcol",
                             keep_cross_validation_predictions = TRUE
)

#' predict the holdout set, in expand data format
#' output is a matrix with 3 columns: predicted label (0/1), prob = 0, prob = 1
test_pred <- h2o.predict(object = model_dl, newdata = test)

#' print out model parameters
print(model_dl@allparameters)
all_para <- model_dl@allparameters

#' the overall performance on training and cross validation (check overfitting)
print(h2o.performance(model_dl, train = T))
print(h2o.performance(model_dl, xval = T))

#' get variable importances
importance_matrix <- as.data.frame(h2o.varimp(model_dl))

#' get cross validation predictions (for stacking, in expand format)
cross_pred <- rep(0, n_train)
h2o_xval_pred <- h2o.cross_validation_predictions(model_dl)
for(i in 1:n_fold) {
  cross_pred <- cross_pred + as.vector(h2o_xval_pred[[i]][, 3])
}
test_pred <- as.vector(test_pred[, 3])

y_pred_prob_feat[1:n_train] <- cross_pred
y_pred_prob_feat[(n_train + 1):n_tot] <- test_pred

#' shutdown h2o
h2o.shutdown(prompt = FALSE)

#' save results from 1st layer model for model stacking
file_name <- paste("dl_result", feature_type, sep = "_")
save(y, y_index, y_pred_prob_feat, allpara, importance_matrix, ind_drop, file = paste(file_name, ".RData", sep = ""))