### DMC 2016 Modeling with rf
library(doParallel)
library(e1071)
library(foreach)

cat('Start generating features...\n')
features <- Features(my_dir = my_dir, binary_response = FALSE, 
                     clean_predictor = FALSE)
cat('Finish generating features...\n')
# or load saved object if you have run once
X <- features$X_train
n_train <- nrow(X)
y <- features$y
n_class <- max(y)+1
y_index <- features$y_index

n_fold = 7
fold_size = trunc(n_train/n_fold)
fold_id = c(rep(1:(n_fold-1), each=fold_size), rep(n_fold, n_train-fold_size*(n_fold-1)))

# train rf

scores = rep(NaN, (n_fold-1))
rate = list(NULL)
X_hold_out = X[fold_id==n_fold, ]
y_hold_out = y[fold_id==n_fold]
y_index_hold_out = y_index[fold_id==n_fold]
X_cv = X[fold_id<n_fold, ]
y_cv = y[fold_id<n_fold]
y_index_cv = y_index[fold_id<n_fold]
fold_id_cv = fold_id[fold_id!=n_fold]
y_pred_sum = matrix(rep(0, n_class*length(y_hold_out)),ncol=n_class)

cluster <- makeCluster(6,"FORK")
registerDoParallel(cluster)
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
  rfParam = expand.grid(mtry=round(sqrt(ncol(X_train))))
  tc<-trainControl(method="none",classProbs=FALSE,allowParallel=TRUE)
  rf = train(X_train,as.factor(y_train),method="parRF",trControl=tc,tuneGrid=rfParam)
  rate[[i]] = varImp(rf) 
  y_cv_pred = predict(rf,X_val,type="prob")
  y_pred = predict(rf,X_hold_out,type="prob")
  if(dim(y_pred)[2]<n_class){ y_pred = cbind(y_pred,matrix(0,nrow=nrow(X_hold_out),ncol=n_class-dim(y_pred)[2]))}
  print(dim(y_pred))

  scores[i] = mean(abs(round(tapply(y_cv_pred[,2], y_index_val, sum))-tapply(y_val, y_index_val, sum)))
  print(dim(y_pred_sum))
  y_pred_sum = y_pred_sum+y_pred
}
closeAllConnections()
y_pred = y_pred_sum/(n_fold-1)
y_pred = round(tapply(y_pred[,2],y_index_hold_out,sum))
y_hold_out = tapply(y_hold_out, y_index_hold_out, sum)


cat('\n')
cat(paste(mean(scores), sd(scores), mean(abs(y_pred-y_hold_out)), '\n'))

# show confusion table
table(y_pred, y_hold_out)

# save data
save(X_cv, y_cv, X_hold_out, y_hold_out, scores, y_pred, y_pred_sum, rate, file = "DMC16.RData")
