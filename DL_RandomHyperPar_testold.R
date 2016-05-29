### deep learning using baseline features
### with parameter tuning
library(h2o)

#setwd("C:/Users/Jihao/Desktop/UCD_DMC16_local")
setwd("~/DMC16/")
#setwd("~/Documents/HaoJi/UC_DAVIS_PHD_in_STAT/2015-2016/DMC16_local/DMC_2016_task_01")
#load(file = "baseFeature_oldSC_noLik.RData") # the baseline feature set (260 columns)
load(file = "expand_data_old.RData") # df_all, y, df_train_quantity
load(file = "DMC16-515-finalfeat.RData") # importance matrix
#load(file = "baseFeature_noWKD.RData")
#load(file = "y_yindex.RData")

# drop orderID since
# (i) its high cardinality;
# (ii) the information it contains can be mostly explained by order date and customer ID.
#df_all = subset(df_all, select=-orderID)
# drop voucherID since voucherAmount is more important
#df_all = subset(df_all, select=-voucherID)
# drop price
#df_all = subset(df_all, select=-price)
# drop orderDate
#df_all = subset(df_all, select=-orderDate)
# drop first orderDate
#df_all = subset(df_all, select=-first_orderDate)
# drop rrp_new
#df_all = subset(df_all, select=-rrp_new)
# drop item_index
#df_all = subset(df_all, select=-c(choice_item_index, ac_item_index, as_item_index, cp_item_index, sp_item_index, csp_item_index, pr_item_index, cpr_item_index, spr_item_index, cspr_item_index, choice_1_item_index, as_1_item_index, sp_1_item_index, pr_1_item_index, spr_1_item_index, choice_234_item_index, as_234_item_index, sp_234_item_index, pr_234_item_index, spr_234_item_index, choice_34_item_index, as_34_item_index, sp_34_item_index, pr_34_item_index, spr_34_item_index, choice_4_item_index, as_4_item_index, sp_4_item_index, pr_4_item_index, spr_4_item_index))
# drop temp
#df_all = subset(df_all, select=-c(temp_ac, temp_as, temp_cp, temp_sp, temp_csp, temp_pr, temp_cpr, temp_spr, temp_cspr, temp_choice_1, temp_as_1, temp_sp_1, temp_pr_1, temp_spr_1, temp_choice_234, temp_as_234, temp_sp_234, temp_pr_234, temp_spr_234, temp_choice_34, temp_as_34, temp_sp_34, temp_pr_34, temp_spr_34, temp_choice_4, temp_as_4, temp_sp_4, temp_pr_4, temp_spr_4))

#OHE_feats = c('order_weekday', 'sizeCode', 'productGroup', 'deviceID', 'paymentMethod')
#df_all = df_all[, !(names(df_all) %in% OHE_feats)]

# expand df_all
df_all = df_all[rep(1:nrow(df_all), times=df_all$quantity), ]
y = rep(rep(c(1,0), length(y)), times = c(t(cbind(y, df_train_quantity-y))))
n_train = length(y)
y_index = rep(1:length(df_train_quantity), times=df_train_quantity)

y = factor(y) # tell DL that it is classification
#X_all = as.matrix(df_all) 
#X_all = df_all # want data frame for DL

### select subset of features for NN execution for speedup
Nfea = 300
imp_var = importance_matrix[1:Nfea, 1]
df_all = df_all[, (names(df_all) %in% imp_var)]
df_all = as.matrix(df_all)
print(paste("most important",ncol(df_all),"features are used.\n",sep=" "))

# block cross-validation
# take the idea from block bootstrap
# 21 months in training data
n_fold = 7
fold_size = trunc(n_train/n_fold)
fold_id = c(rep(1:(n_fold-1), each=fold_size), rep(n_fold, n_train-fold_size*(n_fold-1)))

### split the folds for cross validation
#X_hold_out = X[fold_id==n_fold, ]
#y_hold_out_exp = y[fold_id==n_fold]
#y_index_hold_out = y_index[fold_id==n_fold]
#X_cv = X[fold_id<n_fold, ]
#y_cv = y[fold_id<n_fold]
#y_index_cv = y_index[fold_id<n_fold]
#for(l in likelihood_list){X_hold_out = cbind(X_hold_out, likelihood_test_generator(X_cv, y_cv, X_hold_out, l, expand=T))}
#fold_id_cv = fold_id[fold_id!=n_fold]
#y_pred_sum = rep(0, length(y_hold_out))

### Randomly specifies the parameters for DL
### Make sure that you do not fix any random seed for different execution
hiddenlist = list(c(200,150,100), c(120,90,65,40), c(300,200),
	c(40,40,40,40,40), c(100,100,50,50), c(150,150,150))
Hidden = hiddenlist[[sample(1:length(hiddenlist),1)]]
Hidden_Dropout_Ratios = c(sample(c(0.1,0.25,0.4),1),rep(0,length(Hidden)-1))
L1penalty = sample(c(1e-5,1e-4,1e-3),1)
L2penalty = sample(c(1e-5,1e-4,1e-3),1)

#############################################################
### DEEP LEARNING via H2O
#############################################################
### DL node initialization
h2o.init(startH2O=TRUE, nthreads=-1)
# link data with h2o.init
# test set
test = as.h2o( data.frame(df_all[(n_train+1):nrow(df_all), ]) )

train = as.h2o(data.frame(df_all[1:n_train,],
               response=y, foldcol = fold_id))

# names of the predictors and response
predictors = names(df_all)
response = "response"

### execute DL for Cross Validation, feed data to h2o, as H2OFrame
# automatic CV and model fitting based on whole data using epochs estimated based on CV
CVfold = h2o.deeplearning(x = 1:(ncol(train)-2), y = response, training_frame = train,
                 model_id = "CVFold", # validation set kept empty for automatic CV
                 activation = "RectifierWithDropout", # changable
                 hidden = Hidden, # changable, architecture
                 epochs = 10000,
                 stopping_rounds = 5, # changable
                 stopping_metric = "logloss", # changable
                 stopping_tolerance = 1e-3, # changable
                 # seed = 2016,
                 variable_importances = TRUE,
                 adaptive_rate = TRUE,
                 # rho = 0.99,
                 # epsilon = 1e-7
                 input_dropout_ratio = 0,
                 hidden_dropout_ratios = Hidden_Dropout_Ratios,
                 l1 = L1penalty, # changable
                 l2 = L2penalty, # changable
                 # loss = "Automatic",
                 score_training_samples = 100000,
                 score_validation_samples = 0,
                 score_duty_cycle = 0.1, # 10% at most for scoring validation data
                 max_runtime_secs = 0, # disabled for now
                 missing_values_handling = "MeanImputation", # default
                 #fast_mode = FALSE, # disabled for accuracy
                 # nfolds = 7, # no need once fold_column given
                 fold_column = "foldcol",
                 #fold_assignment = "Modulo",
                 keep_cross_validation_predictions = TRUE
                 )

# predict the holdout set, in expand data format
# output is a matrix with 3 columns: predicted label (0/1), prob=0, prob=1
test_pred = h2o.predict(object=CVfold, newdata=test)
# sum up to original response format for holdout set
#y_pred = round(tapply(as.vector(test_pred[,3]), y_index_hold_out, sum))

# observed hold out response, in original format
#y_hold_out = tapply(as.numeric(y_hold_out_exp)-1, y_index_hold_out, sum)

# print out model parameters
print(CVfold@allparameters)
allpara = CVfold@allparameters
# the overall performance on training and cross validation (check overfitting)
print(h2o.performance(CVfold, train = T))
#score_train = h2o.performance(CVfold, train=T)

print(h2o.performance(CVfold, xval = T))
#score_xval = h2o.performance(CVfold, xval=T)
# overall performance on holdout data
#score_holdout = mean(abs(y_hold_out-y_pred))

# get variable importances
varImp = as.data.frame(h2o.varimp(CVfold))

# get cross validation predictions (for stacking, in expand format)
cross_pred = rep(0, n_train)
h2o_xvalpred = h2o.cross_validation_predictions(CVfold)
for(ii in 1:(n_fold)){
  cross_pred = cross_pred + as.vector(h2o_xvalpred[[ii]][,3])
}
test_pred = as.vector(test_pred[,3])
# in the end, cross_pred and holdout_pred are used for stacking, in expand format

save(test_pred, cross_pred, varImp, allpara, file = "DL_RHP_testold.RData")

h2o.removeAll()
