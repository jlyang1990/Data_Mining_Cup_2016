# first import an old output file and obtain ind_drop (6799 obs)

load("DMC16_stack_xgb_old.RData")
load("DMC16_stack_xgb_new.RData")
load("ind_drop.RData")

y_pred_prob = 0.75*y_pred_prob_old+0.25*y_pred_prob_new
y_pred = round(y_pred_prob)

df_test = read.table(file='/Users/minjay/Documents/Documents/Kaggle/DMC16_Minjie_Fan/DMC_2016_task_01/orders_class.txt',
                     header=TRUE, sep=';')
#df_test <- read.table('orders_class.txt', header=TRUE, sep=';')
results <- df_test[,c("orderID", "articleID", "colorCode", "sizeCode")]
results$prediction <- rep(NA, dim(df_test)[1]) 
results$prediction[ind_drop] <- 0
results$prediction[-ind_drop] = y_pred

write.table(results, file="Uni_UC_Davis_2.txt", sep=";", 
            quote=F,row.names=F)
