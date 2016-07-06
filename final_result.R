#' make the final prediction result for submission
#' written by Qi Gao
#' modified by Jilei Yang

#' load results from model stacking
load("stacking_result_old.RData")
y_pred_prob_old <- y_pred_prob
load("stacking_result_new.RData")
y_pred_prob_new <- y_pred_prob

#' the weights 0.75 and 0.25 are based on our personal beliefs on how reliable the feature transformation for new feature items in testing dataset is
y_pred_prob <- 0.75 * y_pred_prob_old + 0.25 * y_pred_prob_new
y_pred <- round(y_pred_prob)

df_test <- read.table('orders_class.txt', header = TRUE, sep = ';')
results <- df_test[, c("orderID", "articleID", "colorCode", "sizeCode")]
results$prediction <- rep(0, nrow(df_test)) 
results$prediction[-ind_drop] <- y_pred

#' save the final results into txt file
write.table(results, file = "Uni_UC_Davis_2.txt", sep = ";", quote = F, row.names = F)