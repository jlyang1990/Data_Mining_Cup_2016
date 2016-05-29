# second model 
# Data Mining Cup 2016
# written by Minjie Fan, 2016
# modified by Jilei Yang, 2016

rm(list=ls())
library(xgboost)
library(caret)
library(stringr)
library(plyr)
library(sqldf)
library(data.table)
source('Holidays.R')

# load data
df_train = read.table(file='orders_train.txt', header=TRUE, sep=';') 
df_test = read.table(file='orders_class.txt', header=TRUE, sep=';')

# clean data
# drop records with quantity=0 and quantity<returnQuantity
df_train = df_train[df_train$quantity>0 & df_train$quantity>=df_train$returnQuantity, ]
ind_drop = which(df_test$quantity==0)
# drop records with productGroup=NA and rrp=NA 
# since we expect returnQuantity for these records should be zero
# as Qi suggested, these records all have articleID i1004001 and they could be add-on items
# that can not be returned
df_train = df_train[!is.na(df_train$productGroup), ]
ind_drop = c(ind_drop, which(is.na(df_test$productGroup)))
df_test = df_test[-ind_drop, ]

#######################################
# convert new product group and color #
df_test$productGroup[df_test$productGroup==1231] <- 3
df_test$productGroup[df_test$productGroup==1237] <- 2
df_test$productGroup[df_test$productGroup==1289] <- 45

temp = df_test$colorCode[df_test$colorCode>9999]
df_test$colorCode[df_test$colorCode>9999] <- as.integer(temp%%1000+temp%/%10000*1000)
#######################################

n_train = nrow(df_train)
y = df_train$returnQuantity
n_class = max(y)+1
# drop response
df_train = subset(df_train, select=-returnQuantity)

# combine
df_all = rbind(df_train, df_test)

# convert date to year, month, day, weekday
order_ymd = as.data.frame(str_split_fixed(df_all$orderDate, '-', 3))
df_all$order_year = as.numeric(order_ymd[, 1])
df_all$order_month = as.numeric(order_ymd[, 2])
df_all$order_day = as.numeric(order_ymd[, 3])
df_all$order_weekday = weekdays(as.Date(df_all$orderDate))

# OHE
df_all$sizeCode = as.character(df_all$sizeCode)
df_all$productGroup = as.character(df_all$productGroup)
df_all$deviceID = as.character(df_all$deviceID)
df_all$paymentMethod = as.character(df_all$paymentMethod)
OHE_feats = c('order_weekday', 'sizeCode', 'productGroup', 'deviceID', 'paymentMethod')
OHE_formula = paste('~', paste(OHE_feats, collapse = ' + '))
dummies = dummyVars(as.formula(OHE_formula), data = df_all)
df_all_OHE = as.data.frame(predict(dummies, newdata = df_all))
df_all = cbind(df_all, df_all_OHE)

# customerID contains the information of the date when the account was created
# colorCode is already numeric
str2num_feats = c('articleID', 'customerID')
for (f in str2num_feats){
  df_all[, f] = as.numeric(sapply(as.character(df_all[, f]), function(x) substring(x, 3)))
}

# new colorCode
aggr = as.data.frame(t(sapply(df_all$colorCode, function(x) as.numeric(substring(x, first=c(1:4,1,3,1,2), last=c(1:4,2,4,3,4))))))
names(aggr) = c("colorCode_1", "colorCode_2", "colorCode_3", "colorCode_4", "colorCode_12", "colorCode_34", "colorCode_123", "colorCode_234")
df_all = cbind(df_all, aggr)

# new sizeCode
df_all$sizeNew=df_all$sizeCode
df_all[which(df_all$sizeCode=='XS'),]$sizeNew='34'
df_all[which(df_all$sizeCode=='S'),]$sizeNew='36'
df_all[which(df_all$sizeCode=='M'),]$sizeNew='38'
df_all[which(df_all$sizeCode=='L'),]$sizeNew='40'
df_all[which(df_all$sizeCode=='XL'),]$sizeNew='42'
df_all[which(df_all$sizeCode=='75'),]$sizeNew='32'
df_all[which(df_all$sizeCode=='80'),]$sizeNew='34'
df_all[which(df_all$sizeCode=='85'),]$sizeNew='36'
df_all[which(df_all$sizeCode=='90'),]$sizeNew='38'
df_all[which(df_all$sizeCode=='95'),]$sizeNew='40'
df_all[which(df_all$sizeCode=='100'),]$sizeNew='42'

df_all[which(df_all$productGroup=='2' & (df_all$sizeCode %in% c('34'))),]$sizeNew='44'
df_all[which(df_all$productGroup=='2' & (df_all$sizeCode %in% c('33','32'))),]$sizeNew='42'
df_all[which(df_all$productGroup=='2' & (df_all$sizeCode %in% c('33'))),]$sizeNew='42'
df_all[which((df_all$productGroup %in% c('2')) & (df_all$sizeCode %in% c('31','30'))),]$sizeNew='40'
df_all[which((df_all$productGroup %in% c('2')) & (df_all$sizeCode %in% c('29'))),]$sizeNew='38'
df_all[which((df_all$productGroup %in% c('2')) & (df_all$sizeCode %in% c('27','28'))),]$sizeNew='36'
df_all[which((df_all$productGroup %in% c('2')) & (df_all$sizeCode %in% c('26'))),]$sizeNew='34'
df_all[which((df_all$productGroup %in% c('2')) & (df_all$sizeCode %in% c('25','24'))),]$sizeNew='32'

df_all[which(df_all$sizeCode=='A'),]$sizeNew = NA
df_all[which(df_all$sizeCode=='I'),]$sizeNew = NA

df_all$sizeNew = as.numeric(df_all$sizeNew)
df_all$sizeNew[is.na(df_all$sizeNew)] = mean(df_all$sizeNew, na.rm = TRUE)

# price per quantity: price/quantity
df_all$price_per_quantity = df_all$price/df_all$quantity
# discounted price per quantity: rrp-price_per_quantity
df_all$discounted_price_per_quantity = df_all$rrp-df_all$price_per_quantity
# discounted ratio per quantity
df_all$discounted_ratio_per_quantity = df_all$discounted_price_per_quantity/df_all$rrp
df_all$discounted_ratio_per_quantity[is.na(df_all$discounted_ratio_per_quantity)] = 0
# true price per quantity
df_all$true_price_per_quantity = df_all$price_per_quantity*(1-df_all$voucherAmount/tapply(df_all$price, df_all$orderID, sum)[df_all$orderID])
df_all$true_price_per_quantity[is.na(df_all$true_price_per_quantity)] = 0
# true discounted price per quantity
df_all$true_discounted_price_per_quantity = df_all$rrp-df_all$true_price_per_quantity
# true discounted ratio per quantity
df_all$true_discounted_ratio_per_quantity = df_all$true_discounted_price_per_quantity/df_all$rrp
df_all$true_discounted_ratio_per_quantity[is.na(df_all$true_discounted_ratio_per_quantity)] = 0

# indices of whether voucherID exists (there are some cases where voucherID exists but voucherAmount is 0)
#df_all$voucherID_index = as.numeric(df_all$voucherID!="0")

# order feature ###############################################################################################################
# total frequency of order items per order
df_all$freq_per_order = tapply(df_all$quantity, df_all$orderID, length)[df_all$orderID]
# total quantity per order
df_all$quantity_per_order = tapply(df_all$quantity, df_all$orderID, sum)[df_all$orderID]
# total number of types of articleID per order
df_all$article_per_order = tapply(df_all$articleID, df_all$orderID, function(x) length(unique(x)))[df_all$orderID]
# total number of types of productGroup per order
df_all$prod_per_order = tapply(df_all$productGroup, df_all$orderID, function(x) length(unique(x)))[df_all$orderID]
# total number of types of colorCode per order
df_all$color_per_order = tapply(df_all$colorCode, df_all$orderID, function(x) length(unique(x)))[df_all$orderID]
# total number of types of sizeCode per order
df_all$size_per_order = tapply(as.character(df_all$sizeCode), df_all$orderID, function(x) length(unique(x)))[df_all$orderID]

# total original price per order
df_all$original_price_per_order = tapply(df_all$rrp*df_all$quantity, df_all$orderID, sum)[df_all$orderID]
# total price per order
df_all$price_per_order = tapply(df_all$price, df_all$orderID, sum)[df_all$orderID]
# total true price per order
df_all$true_price_per_order = tapply(df_all$price, df_all$orderID, sum)[df_all$orderID] - df_all$voucherAmount
# total discounted price per order
df_all$discounted_price_per_order = df_all$original_price_per_order - df_all$true_price_per_order
# total discounted ratio per order
df_all$discounted_ratio_per_order = df_all$discounted_price_per_order/df_all$original_price_per_order
df_all$discounted_ratio_per_order[is.na(df_all$discounted_ratio_per_order)] = 0
# mean, max, min rrp per order
df_all$mean_rrp_per_order = tapply(df_all$rrp, df_all$orderID, mean)[df_all$orderID]
df_all$max_rrp_per_order = tapply(df_all$rrp, df_all$orderID, max)[df_all$orderID]
df_all$min_rrp_per_order = tapply(df_all$rrp, df_all$orderID, min)[df_all$orderID]

# choice order item: items with the same articleID in a single order with different colorCode/sizeCode
# indices of choice order item (1: yes, 0: no)
df_all$choice_item_index = as.numeric(unlist(tapply(df_all$articleID, df_all$orderID, function(x) duplicated(x)|duplicated(x, fromLast=T))[unique(df_all$orderID)]))
aggr = aggregate(rep(1,nrow(df_all)), list(df_all$articleID, df_all$orderID), sum)
names(aggr) = c("articleID","orderID","choice_item_quantity_index")
df_all=join(df_all, aggr, by=c("articleID","orderID"))
# total number of types of choice order items per order
df_all$choice_item_per_order = df_all$article_per_order - tapply(1-df_all$choice_item_index, df_all$orderID, sum)[df_all$orderID]
# ratio of types of choice order items per order
df_all$choice_item_ratio_per_order = df_all$choice_item_per_order/df_all$article_per_order
# total quantity of choice order items per order
df_all$choice_item_quantity_per_order = tapply(df_all$quantity*df_all$choice_item_index, df_all$orderID, sum)[df_all$orderID]
# ratio of quantity of choice order items per order
df_all$choice_item_quantity_ratio_per_order = df_all$choice_item_quantity_per_order/df_all$quantity_per_order

# ac item: items with the same articleID and colorCode in a single order
df_all$temp_ac = apply(cbind(df_all$articleID, df_all$colorCode), 1, function(x) paste(x, sep="", collapse=" "))
df_all$ac_item_index = as.numeric(unlist(tapply(df_all$temp_ac, df_all$orderID, function(x) duplicated(x)|duplicated(x, fromLast=T))[unique(df_all$orderID)]))
aggr = aggregate(rep(1,nrow(df_all)), list(df_all$temp_ac, df_all$orderID), sum)
names(aggr) = c("temp_ac","orderID","ac_item_quantity_index")
df_all=join(df_all, aggr, by=c("temp_ac","orderID"))
df_all$ac_item_quantity_per_order = tapply(df_all$quantity*df_all$ac_item_index, df_all$orderID, sum)[df_all$orderID]
df_all$ac_item_quantity_ratio_per_order = df_all$ac_item_quantity_per_order/df_all$quantity_per_order

# as item: items with the same articleID and sizeCode in a single order
df_all$temp_as = apply(cbind(df_all$articleID, df_all$sizeCode), 1, function(x) paste(x, sep="", collapse=" "))
df_all$as_item_index = as.numeric(unlist(tapply(df_all$temp_as, df_all$orderID, function(x) duplicated(x)|duplicated(x, fromLast=T))[unique(df_all$orderID)]))
aggr = aggregate(rep(1,nrow(df_all)), list(df_all$temp_as, df_all$orderID), sum)
names(aggr) = c("temp_as","orderID","as_item_quantity_index")
df_all=join(df_all, aggr, by=c("temp_as","orderID"))
df_all$as_item_quantity_per_order = tapply(df_all$quantity*df_all$as_item_index, df_all$orderID, sum)[df_all$orderID]
df_all$as_item_quantity_ratio_per_order = df_all$as_item_quantity_per_order/df_all$quantity_per_order

# cp item: items with the same productGroup and colorCode in a single order
df_all$temp_cp = apply(cbind(df_all$colorCode, df_all$productGroup), 1, function(x) paste(x, sep="", collapse=" "))
df_all$cp_item_index = as.numeric(unlist(tapply(df_all$temp_cp, df_all$orderID, function(x) duplicated(x)|duplicated(x, fromLast=T))[unique(df_all$orderID)]))
aggr = aggregate(rep(1,nrow(df_all)), list(df_all$temp_cp, df_all$orderID), sum)
names(aggr) = c("temp_cp","orderID","cp_item_quantity_index")
df_all=join(df_all, aggr, by=c("temp_cp","orderID"))
df_all$cp_item_quantity_per_order = tapply(df_all$quantity*df_all$cp_item_index, df_all$orderID, sum)[df_all$orderID]
df_all$cp_item_quantity_ratio_per_order = df_all$cp_item_quantity_per_order/df_all$quantity_per_order

# sp item: items with the same productGroup and sizeCode in a single order
df_all$temp_sp = apply(cbind(as.character(df_all$sizeCode), df_all$productGroup), 1, function(x) paste(x, sep="", collapse=" "))
df_all$sp_item_index = as.numeric(unlist(tapply(df_all$temp_sp, df_all$orderID, function(x) duplicated(x)|duplicated(x, fromLast=T))[unique(df_all$orderID)]))
aggr = aggregate(rep(1,nrow(df_all)), list(df_all$temp_sp, df_all$orderID), sum)
names(aggr) = c("temp_sp","orderID","sp_item_quantity_index")
df_all=join(df_all, aggr, by=c("temp_sp","orderID"))
df_all$sp_item_quantity_per_order = tapply(df_all$quantity*df_all$sp_item_index, df_all$orderID, sum)[df_all$orderID]
df_all$sp_item_quantity_ratio_per_order = df_all$sp_item_quantity_per_order/df_all$quantity_per_order

# csp item: items with the same productGroup, colorCode and sizeCode in a single order
df_all$temp_csp = apply(cbind(df_all$colorCode, as.character(df_all$sizeCode), df_all$productGroup), 1, function(x) paste(x, sep="", collapse=" "))
df_all$csp_item_index = as.numeric(unlist(tapply(df_all$temp_csp, df_all$orderID, function(x) duplicated(x)|duplicated(x, fromLast=T))[unique(df_all$orderID)]))
aggr = aggregate(rep(1,nrow(df_all)), list(df_all$temp_csp, df_all$orderID), sum)
names(aggr) = c("temp_csp","orderID","csp_item_quantity_index")
df_all=join(df_all, aggr, by=c("temp_csp","orderID"))
df_all$csp_item_quantity_per_order = tapply(df_all$quantity*df_all$csp_item_index, df_all$orderID, sum)[df_all$orderID]
df_all$csp_item_quantity_ratio_per_order = df_all$csp_item_quantity_per_order/df_all$quantity_per_order

df_all$rrp_new = df_all$rrp
df_all$rrp_new[df_all$rrp_new==9.95] = 9.99
df_all$rrp_new[df_all$rrp_new==12.95] = 12.99
df_all$rrp_new[df_all$rrp_new %in% c(15, 15.95)] = 15.99
df_all$rrp_new[df_all$rrp_new==19.95] = 19.99
df_all$rrp_new[df_all$rrp_new==29.95] = 29.99
df_all$rrp_new[df_all$rrp_new==35] = 35.99
df_all$rrp_new[df_all$rrp_new %in% c(39.95, 40)] = 39.99
df_all$rrp_new[df_all$rrp_new==49.95] = 49.99
df_all$rrp_new[df_all$rrp_new==59.95] = 59.99
df_all$rrp_new[df_all$rrp_new==69.95] = 69.99

# pr item: items with the same rrp and productGroup in a single order
df_all$temp_pr = apply(cbind(df_all$productGroup, df_all$rrp_new), 1, function(x) paste(x, sep="", collapse=" "))
df_all$pr_item_index = as.numeric(unlist(tapply(df_all$temp_pr, df_all$orderID, function(x) duplicated(x)|duplicated(x, fromLast=T))[unique(df_all$orderID)]))
aggr = aggregate(rep(1,nrow(df_all)), list(df_all$temp_pr, df_all$orderID), sum)
names(aggr) = c("temp_pr","orderID","pr_item_quantity_index")
df_all=join(df_all, aggr, by=c("temp_pr","orderID"))
df_all$pr_item_quantity_per_order = tapply(df_all$quantity*df_all$pr_item_index, df_all$orderID, sum)[df_all$orderID]
df_all$pr_item_quantity_ratio_per_order = df_all$pr_item_quantity_per_order/df_all$quantity_per_order

# cpr item: items with the same rrp, productGroup and colorCode in a single order
df_all$temp_cpr = apply(cbind(df_all$colorCode, df_all$productGroup, df_all$rrp_new), 1, function(x) paste(x, sep="", collapse=" "))
df_all$cpr_item_index = as.numeric(unlist(tapply(df_all$temp_cpr, df_all$orderID, function(x) duplicated(x)|duplicated(x, fromLast=T))[unique(df_all$orderID)]))
aggr = aggregate(rep(1,nrow(df_all)), list(df_all$temp_cpr, df_all$orderID), sum)
names(aggr) = c("temp_cpr","orderID","cpr_item_quantity_index")
df_all=join(df_all, aggr, by=c("temp_cpr","orderID"))
df_all$cpr_item_quantity_per_order = tapply(df_all$quantity*df_all$cpr_item_index, df_all$orderID, sum)[df_all$orderID]
df_all$cpr_item_quantity_ratio_per_order = df_all$cpr_item_quantity_per_order/df_all$quantity_per_order

# spr item: items with the same rrp, productGroup and sizeCode in a single order
df_all$temp_spr = apply(cbind(as.character(df_all$sizeCode), df_all$productGroup, df_all$rrp_new), 1, function(x) paste(x, sep="", collapse=" "))
df_all$spr_item_index = as.numeric(unlist(tapply(df_all$temp_spr, df_all$orderID, function(x) duplicated(x)|duplicated(x, fromLast=T))[unique(df_all$orderID)]))
aggr = aggregate(rep(1,nrow(df_all)), list(df_all$temp_spr, df_all$orderID), sum)
names(aggr) = c("temp_spr","orderID","spr_item_quantity_index")
df_all=join(df_all, aggr, by=c("temp_spr","orderID"))
df_all$spr_item_quantity_per_order = tapply(df_all$quantity*df_all$spr_item_index, df_all$orderID, sum)[df_all$orderID]
df_all$spr_item_quantity_ratio_per_order = df_all$spr_item_quantity_per_order/df_all$quantity_per_order

# cspr item: items with the same rrp, productGroup, colorCode and sizeCode in a single order
df_all$temp_cspr = apply(cbind(df_all$colorCode, as.character(df_all$sizeCode), df_all$productGroup, df_all$rrp_new), 1, function(x) paste(x, sep="", collapse=" "))
df_all$cspr_item_index = as.numeric(unlist(tapply(df_all$temp_cspr, df_all$orderID, function(x) duplicated(x)|duplicated(x, fromLast=T))[unique(df_all$orderID)]))
aggr = aggregate(rep(1,nrow(df_all)), list(df_all$temp_cspr, df_all$orderID), sum)
names(aggr) = c("temp_cspr","orderID","cspr_item_quantity_index")
df_all=join(df_all, aggr, by=c("temp_cspr","orderID"))
df_all$cspr_item_quantity_per_order = tapply(df_all$quantity*df_all$cspr_item_index, df_all$orderID, sum)[df_all$orderID]
df_all$cspr_item_quantity_ratio_per_order = df_all$cspr_item_quantity_per_order/df_all$quantity_per_order

# choice_1 item: items with the same articleID in a single order
df_all$temp_choice_1 = apply(cbind(df_all$articleID, df_all$colorCode_1), 1, function(x) paste(x, sep="", collapse=" "))
df_all$choice_1_item_index = as.numeric(unlist(tapply(df_all$temp_choice_1, df_all$orderID, function(x) duplicated(x)|duplicated(x, fromLast=T))[unique(df_all$orderID)]))
aggr = aggregate(rep(1,nrow(df_all)), list(df_all$temp_choice_1, df_all$orderID), sum)
names(aggr) = c("temp_choice_1","orderID","choice_1_item_quantity_index")
df_all=join(df_all, aggr, by=c("temp_choice_1","orderID"))
df_all$choice_1_item_quantity_per_order = tapply(df_all$quantity*df_all$choice_1_item_index, df_all$orderID, sum)[df_all$orderID]
df_all$choice_1_item_quantity_ratio_per_order = df_all$choice_1_item_quantity_per_order/df_all$quantity_per_order

# as_1 item: items with the same articleID and sizeCode in a single order
df_all$temp_as_1 = apply(cbind(df_all$articleID, df_all$sizeCode, df_all$colorCode_1), 1, function(x) paste(x, sep="", collapse=" "))
df_all$as_1_item_index = as.numeric(unlist(tapply(df_all$temp_as_1, df_all$orderID, function(x) duplicated(x)|duplicated(x, fromLast=T))[unique(df_all$orderID)]))
aggr = aggregate(rep(1,nrow(df_all)), list(df_all$temp_as_1, df_all$orderID), sum)
names(aggr) = c("temp_as_1","orderID","as_1_item_quantity_index")
df_all=join(df_all, aggr, by=c("temp_as_1","orderID"))
df_all$as_1_item_quantity_per_order = tapply(df_all$quantity*df_all$as_1_item_index, df_all$orderID, sum)[df_all$orderID]
df_all$as_1_item_quantity_ratio_per_order = df_all$as_1_item_quantity_per_order/df_all$quantity_per_order

# sp_1 item: items with the same productGroup and sizeCode in a single order
df_all$temp_sp_1 = apply(cbind(as.character(df_all$sizeCode), df_all$productGroup, df_all$colorCode_1), 1, function(x) paste(x, sep="", collapse=" "))
df_all$sp_1_item_index = as.numeric(unlist(tapply(df_all$temp_sp_1, df_all$orderID, function(x) duplicated(x)|duplicated(x, fromLast=T))[unique(df_all$orderID)]))
aggr = aggregate(rep(1,nrow(df_all)), list(df_all$temp_sp_1, df_all$orderID), sum)
names(aggr) = c("temp_sp_1","orderID","sp_1_item_quantity_index")
df_all=join(df_all, aggr, by=c("temp_sp_1","orderID"))
df_all$sp_1_item_quantity_per_order = tapply(df_all$quantity*df_all$sp_1_item_index, df_all$orderID, sum)[df_all$orderID]
df_all$sp_1_item_quantity_ratio_per_order = df_all$sp_1_item_quantity_per_order/df_all$quantity_per_order

# pr_1 item: items with the same rrp and productGroup in a single order
df_all$temp_pr_1 = apply(cbind(as.character(df_all$sizeCode), df_all$productGroup, df_all$rrp_new, df_all$colorCode_1), 1, function(x) paste(x, sep="", collapse=" "))
df_all$pr_1_item_index = as.numeric(unlist(tapply(df_all$temp_pr_1, df_all$orderID, function(x) duplicated(x)|duplicated(x, fromLast=T))[unique(df_all$orderID)]))
aggr = aggregate(rep(1,nrow(df_all)), list(df_all$temp_pr_1, df_all$orderID), sum)
names(aggr) = c("temp_pr_1","orderID","pr_1_item_quantity_index")
df_all=join(df_all, aggr, by=c("temp_pr_1","orderID"))
df_all$pr_1_item_quantity_per_order = tapply(df_all$quantity*df_all$pr_1_item_index, df_all$orderID, sum)[df_all$orderID]
df_all$pr_1_item_quantity_ratio_per_order = df_all$pr_1_item_quantity_per_order/df_all$quantity_per_order

# spr_1 item: items with the same rrp, productGroup and sizeCode in a single order
df_all$temp_spr_1 = apply(cbind(as.character(df_all$sizeCode), df_all$productGroup, df_all$rrp_new, df_all$colorCode_1), 1, function(x) paste(x, sep="", collapse=" "))
df_all$spr_1_item_index = as.numeric(unlist(tapply(df_all$temp_spr_1, df_all$orderID, function(x) duplicated(x)|duplicated(x, fromLast=T))[unique(df_all$orderID)]))
aggr = aggregate(rep(1,nrow(df_all)), list(df_all$temp_spr_1, df_all$orderID), sum)
names(aggr) = c("temp_spr_1","orderID","spr_1_item_quantity_index")
df_all=join(df_all, aggr, by=c("temp_spr_1","orderID"))
df_all$spr_1_item_quantity_per_order = tapply(df_all$quantity*df_all$spr_1_item_index, df_all$orderID, sum)[df_all$orderID]
df_all$spr_1_item_quantity_ratio_per_order = df_all$spr_1_item_quantity_per_order/df_all$quantity_per_order

# choice_234 item: items with the same articleID in a single order
df_all$temp_choice_234 = apply(cbind(df_all$articleID, df_all$colorCode_234), 1, function(x) paste(x, sep="", collapse=" "))
df_all$choice_234_item_index = as.numeric(unlist(tapply(df_all$temp_choice_234, df_all$orderID, function(x) duplicated(x)|duplicated(x, fromLast=T))[unique(df_all$orderID)]))
aggr = aggregate(rep(1,nrow(df_all)), list(df_all$temp_choice_234, df_all$orderID), sum)
names(aggr) = c("temp_choice_234","orderID","choice_234_item_quantity_index")
df_all=join(df_all, aggr, by=c("temp_choice_234","orderID"))
df_all$choice_234_item_quantity_per_order = tapply(df_all$quantity*df_all$choice_234_item_index, df_all$orderID, sum)[df_all$orderID]
df_all$choice_234_item_quantity_ratio_per_order = df_all$choice_234_item_quantity_per_order/df_all$quantity_per_order

# as_234 item: items with the same articleID and sizeCode in a single order
df_all$temp_as_234 = apply(cbind(df_all$articleID, df_all$sizeCode, df_all$colorCode_234), 1, function(x) paste(x, sep="", collapse=" "))
df_all$as_234_item_index = as.numeric(unlist(tapply(df_all$temp_as_234, df_all$orderID, function(x) duplicated(x)|duplicated(x, fromLast=T))[unique(df_all$orderID)]))
aggr = aggregate(rep(1,nrow(df_all)), list(df_all$temp_as_234, df_all$orderID), sum)
names(aggr) = c("temp_as_234","orderID","as_234_item_quantity_index")
df_all=join(df_all, aggr, by=c("temp_as_234","orderID"))
df_all$as_234_item_quantity_per_order = tapply(df_all$quantity*df_all$as_234_item_index, df_all$orderID, sum)[df_all$orderID]
df_all$as_234_item_quantity_ratio_per_order = df_all$as_234_item_quantity_per_order/df_all$quantity_per_order

# sp_234 item: items with the same productGroup and sizeCode in a single order
df_all$temp_sp_234 = apply(cbind(as.character(df_all$sizeCode), df_all$productGroup, df_all$colorCode_234), 1, function(x) paste(x, sep="", collapse=" "))
df_all$sp_234_item_index = as.numeric(unlist(tapply(df_all$temp_sp_234, df_all$orderID, function(x) duplicated(x)|duplicated(x, fromLast=T))[unique(df_all$orderID)]))
aggr = aggregate(rep(1,nrow(df_all)), list(df_all$temp_sp_234, df_all$orderID), sum)
names(aggr) = c("temp_sp_234","orderID","sp_234_item_quantity_index")
df_all=join(df_all, aggr, by=c("temp_sp_234","orderID"))
df_all$sp_234_item_quantity_per_order = tapply(df_all$quantity*df_all$sp_234_item_index, df_all$orderID, sum)[df_all$orderID]
df_all$sp_234_item_quantity_ratio_per_order = df_all$sp_234_item_quantity_per_order/df_all$quantity_per_order

# pr_234 item: items with the same rrp and productGroup in a single order
df_all$temp_pr_234 = apply(cbind(as.character(df_all$sizeCode), df_all$productGroup, df_all$rrp_new, df_all$colorCode_234), 1, function(x) paste(x, sep="", collapse=" "))
df_all$pr_234_item_index = as.numeric(unlist(tapply(df_all$temp_pr_234, df_all$orderID, function(x) duplicated(x)|duplicated(x, fromLast=T))[unique(df_all$orderID)]))
aggr = aggregate(rep(1,nrow(df_all)), list(df_all$temp_pr_234, df_all$orderID), sum)
names(aggr) = c("temp_pr_234","orderID","pr_234_item_quantity_index")
df_all=join(df_all, aggr, by=c("temp_pr_234","orderID"))
df_all$pr_234_item_quantity_per_order = tapply(df_all$quantity*df_all$pr_234_item_index, df_all$orderID, sum)[df_all$orderID]
df_all$pr_234_item_quantity_ratio_per_order = df_all$pr_234_item_quantity_per_order/df_all$quantity_per_order

# spr_234 item: items with the same rrp, productGroup and sizeCode in a single order
df_all$temp_spr_234 = apply(cbind(as.character(df_all$sizeCode), df_all$productGroup, df_all$rrp_new, df_all$colorCode_234), 1, function(x) paste(x, sep="", collapse=" "))
df_all$spr_234_item_index = as.numeric(unlist(tapply(df_all$temp_spr_234, df_all$orderID, function(x) duplicated(x)|duplicated(x, fromLast=T))[unique(df_all$orderID)]))
aggr = aggregate(rep(1,nrow(df_all)), list(df_all$temp_spr_234, df_all$orderID), sum)
names(aggr) = c("temp_spr_234","orderID","spr_234_item_quantity_index")
df_all=join(df_all, aggr, by=c("temp_spr_234","orderID"))
df_all$spr_234_item_quantity_per_order = tapply(df_all$quantity*df_all$spr_234_item_index, df_all$orderID, sum)[df_all$orderID]
df_all$spr_234_item_quantity_ratio_per_order = df_all$spr_234_item_quantity_per_order/df_all$quantity_per_order

# choice_34 item: items with the same articleID in a single order
df_all$temp_choice_34 = apply(cbind(df_all$articleID, df_all$colorCode_34), 1, function(x) paste(x, sep="", collapse=" "))
df_all$choice_34_item_index = as.numeric(unlist(tapply(df_all$temp_choice_34, df_all$orderID, function(x) duplicated(x)|duplicated(x, fromLast=T))[unique(df_all$orderID)]))
aggr = aggregate(rep(1,nrow(df_all)), list(df_all$temp_choice_34, df_all$orderID), sum)
names(aggr) = c("temp_choice_34","orderID","choice_34_item_quantity_index")
df_all=join(df_all, aggr, by=c("temp_choice_34","orderID"))
df_all$choice_34_item_quantity_per_order = tapply(df_all$quantity*df_all$choice_34_item_index, df_all$orderID, sum)[df_all$orderID]
df_all$choice_34_item_quantity_ratio_per_order = df_all$choice_34_item_quantity_per_order/df_all$quantity_per_order

# as_34 item: items with the same articleID and sizeCode in a single order
df_all$temp_as_34 = apply(cbind(df_all$articleID, df_all$sizeCode, df_all$colorCode_34), 1, function(x) paste(x, sep="", collapse=" "))
df_all$as_34_item_index = as.numeric(unlist(tapply(df_all$temp_as_34, df_all$orderID, function(x) duplicated(x)|duplicated(x, fromLast=T))[unique(df_all$orderID)]))
aggr = aggregate(rep(1,nrow(df_all)), list(df_all$temp_as_34, df_all$orderID), sum)
names(aggr) = c("temp_as_34","orderID","as_34_item_quantity_index")
df_all=join(df_all, aggr, by=c("temp_as_34","orderID"))
df_all$as_34_item_quantity_per_order = tapply(df_all$quantity*df_all$as_34_item_index, df_all$orderID, sum)[df_all$orderID]
df_all$as_34_item_quantity_ratio_per_order = df_all$as_34_item_quantity_per_order/df_all$quantity_per_order

# sp_34 item: items with the same productGroup and sizeCode in a single order
df_all$temp_sp_34 = apply(cbind(as.character(df_all$sizeCode), df_all$productGroup, df_all$colorCode_34), 1, function(x) paste(x, sep="", collapse=" "))
df_all$sp_34_item_index = as.numeric(unlist(tapply(df_all$temp_sp_34, df_all$orderID, function(x) duplicated(x)|duplicated(x, fromLast=T))[unique(df_all$orderID)]))
aggr = aggregate(rep(1,nrow(df_all)), list(df_all$temp_sp_34, df_all$orderID), sum)
names(aggr) = c("temp_sp_34","orderID","sp_34_item_quantity_index")
df_all=join(df_all, aggr, by=c("temp_sp_34","orderID"))
df_all$sp_34_item_quantity_per_order = tapply(df_all$quantity*df_all$sp_34_item_index, df_all$orderID, sum)[df_all$orderID]
df_all$sp_34_item_quantity_ratio_per_order = df_all$sp_34_item_quantity_per_order/df_all$quantity_per_order

# pr_34 item: items with the same rrp and productGroup in a single order
df_all$temp_pr_34 = apply(cbind(as.character(df_all$sizeCode), df_all$productGroup, df_all$rrp_new, df_all$colorCode_34), 1, function(x) paste(x, sep="", collapse=" "))
df_all$pr_34_item_index = as.numeric(unlist(tapply(df_all$temp_pr_34, df_all$orderID, function(x) duplicated(x)|duplicated(x, fromLast=T))[unique(df_all$orderID)]))
aggr = aggregate(rep(1,nrow(df_all)), list(df_all$temp_pr_34, df_all$orderID), sum)
names(aggr) = c("temp_pr_34","orderID","pr_34_item_quantity_index")
df_all=join(df_all, aggr, by=c("temp_pr_34","orderID"))
df_all$pr_34_item_quantity_per_order = tapply(df_all$quantity*df_all$pr_34_item_index, df_all$orderID, sum)[df_all$orderID]
df_all$pr_34_item_quantity_ratio_per_order = df_all$pr_34_item_quantity_per_order/df_all$quantity_per_order

# spr_34 item: items with the same rrp, productGroup and sizeCode in a single order
df_all$temp_spr_34 = apply(cbind(as.character(df_all$sizeCode), df_all$productGroup, df_all$rrp_new, df_all$colorCode_34), 1, function(x) paste(x, sep="", collapse=" "))
df_all$spr_34_item_index = as.numeric(unlist(tapply(df_all$temp_spr_34, df_all$orderID, function(x) duplicated(x)|duplicated(x, fromLast=T))[unique(df_all$orderID)]))
aggr = aggregate(rep(1,nrow(df_all)), list(df_all$temp_spr_34, df_all$orderID), sum)
names(aggr) = c("temp_spr_34","orderID","spr_34_item_quantity_index")
df_all=join(df_all, aggr, by=c("temp_spr_34","orderID"))
df_all$spr_34_item_quantity_per_order = tapply(df_all$quantity*df_all$spr_34_item_index, df_all$orderID, sum)[df_all$orderID]
df_all$spr_34_item_quantity_ratio_per_order = df_all$spr_34_item_quantity_per_order/df_all$quantity_per_order

# choice_4 item: items with the same articleID in a single order
df_all$temp_choice_4 = apply(cbind(df_all$articleID, df_all$colorCode_4), 1, function(x) paste(x, sep="", collapse=" "))
df_all$choice_4_item_index = as.numeric(unlist(tapply(df_all$temp_choice_4, df_all$orderID, function(x) duplicated(x)|duplicated(x, fromLast=T))[unique(df_all$orderID)]))
aggr = aggregate(rep(1,nrow(df_all)), list(df_all$temp_choice_4, df_all$orderID), sum)
names(aggr) = c("temp_choice_4","orderID","choice_4_item_quantity_index")
df_all=join(df_all, aggr, by=c("temp_choice_4","orderID"))
df_all$choice_4_item_quantity_per_order = tapply(df_all$quantity*df_all$choice_4_item_index, df_all$orderID, sum)[df_all$orderID]
df_all$choice_4_item_quantity_ratio_per_order = df_all$choice_4_item_quantity_per_order/df_all$quantity_per_order

# as_4 item: items with the same articleID and sizeCode in a single order
df_all$temp_as_4 = apply(cbind(df_all$articleID, df_all$sizeCode, df_all$colorCode_4), 1, function(x) paste(x, sep="", collapse=" "))
df_all$as_4_item_index = as.numeric(unlist(tapply(df_all$temp_as_4, df_all$orderID, function(x) duplicated(x)|duplicated(x, fromLast=T))[unique(df_all$orderID)]))
aggr = aggregate(rep(1,nrow(df_all)), list(df_all$temp_as_4, df_all$orderID), sum)
names(aggr) = c("temp_as_4","orderID","as_4_item_quantity_index")
df_all=join(df_all, aggr, by=c("temp_as_4","orderID"))
df_all$as_4_item_quantity_per_order = tapply(df_all$quantity*df_all$as_4_item_index, df_all$orderID, sum)[df_all$orderID]
df_all$as_4_item_quantity_ratio_per_order = df_all$as_4_item_quantity_per_order/df_all$quantity_per_order

# sp_4 item: items with the same productGroup and sizeCode in a single order
df_all$temp_sp_4 = apply(cbind(as.character(df_all$sizeCode), df_all$productGroup, df_all$colorCode_4), 1, function(x) paste(x, sep="", collapse=" "))
df_all$sp_4_item_index = as.numeric(unlist(tapply(df_all$temp_sp_4, df_all$orderID, function(x) duplicated(x)|duplicated(x, fromLast=T))[unique(df_all$orderID)]))
aggr = aggregate(rep(1,nrow(df_all)), list(df_all$temp_sp_4, df_all$orderID), sum)
names(aggr) = c("temp_sp_4","orderID","sp_4_item_quantity_index")
df_all=join(df_all, aggr, by=c("temp_sp_4","orderID"))
df_all$sp_4_item_quantity_per_order = tapply(df_all$quantity*df_all$sp_4_item_index, df_all$orderID, sum)[df_all$orderID]
df_all$sp_4_item_quantity_ratio_per_order = df_all$sp_4_item_quantity_per_order/df_all$quantity_per_order

# pr_4 item: items with the same rrp and productGroup in a single order
df_all$temp_pr_4 = apply(cbind(as.character(df_all$sizeCode), df_all$productGroup, df_all$rrp_new, df_all$colorCode_4), 1, function(x) paste(x, sep="", collapse=" "))
df_all$pr_4_item_index = as.numeric(unlist(tapply(df_all$temp_pr_4, df_all$orderID, function(x) duplicated(x)|duplicated(x, fromLast=T))[unique(df_all$orderID)]))
aggr = aggregate(rep(1,nrow(df_all)), list(df_all$temp_pr_4, df_all$orderID), sum)
names(aggr) = c("temp_pr_4","orderID","pr_4_item_quantity_index")
df_all=join(df_all, aggr, by=c("temp_pr_4","orderID"))
df_all$pr_4_item_quantity_per_order = tapply(df_all$quantity*df_all$pr_4_item_index, df_all$orderID, sum)[df_all$orderID]
df_all$pr_4_item_quantity_ratio_per_order = df_all$pr_4_item_quantity_per_order/df_all$quantity_per_order

# spr_4 item: items with the same rrp, productGroup and sizeCode in a single order
df_all$temp_spr_4 = apply(cbind(as.character(df_all$sizeCode), df_all$productGroup, df_all$rrp_new, df_all$colorCode_4), 1, function(x) paste(x, sep="", collapse=" "))
df_all$spr_4_item_index = as.numeric(unlist(tapply(df_all$temp_spr_4, df_all$orderID, function(x) duplicated(x)|duplicated(x, fromLast=T))[unique(df_all$orderID)]))
aggr = aggregate(rep(1,nrow(df_all)), list(df_all$temp_spr_4, df_all$orderID), sum)
names(aggr) = c("temp_spr_4","orderID","spr_4_item_quantity_index")
df_all=join(df_all, aggr, by=c("temp_spr_4","orderID"))
df_all$spr_4_item_quantity_per_order = tapply(df_all$quantity*df_all$spr_4_item_index, df_all$orderID, sum)[df_all$orderID]
df_all$spr_4_item_quantity_ratio_per_order = df_all$spr_4_item_quantity_per_order/df_all$quantity_per_order

# customer feature ############################################################################################################
# total number of orders per customer
df_all$order_per_customer = tapply(as.character(df_all$orderID), df_all$customerID, function(x) length(unique(x)))[as.character(df_all$customerID)]
# total frequency of order items per customer
df_all$freq_per_customer = tapply(df_all$quantity, df_all$customerID, length)[as.character(df_all$customerID)]
# total quantity per customer
df_all$quantity_per_customer = tapply(df_all$quantity, df_all$customerID, sum)[as.character(df_all$customerID)]
# total number of types of articleID per customer
df_all$article_per_customer = tapply(df_all$articleID, df_all$customerID, function(x) length(unique(x)))[as.character(df_all$customerID)]
# total number of types of productGroup per customer
df_all$prod_per_customer = tapply(df_all$productGroup, df_all$customerID, function(x) length(unique(x)))[as.character(df_all$customerID)]
# total number of types of colorCode per customer
df_all$colorCode_per_customer = tapply(df_all$colorCode, df_all$customerID, function(x) length(unique(x)))[as.character(df_all$customerID)]
# total number of types of sizeCode per customer
df_all$sizeCode_per_customer = tapply(as.character(df_all$sizeCode), df_all$customerID, function(x) length(unique(x)))[as.character(df_all$customerID)]
# total number of types of deviceID per customer
df_all$device_per_customer = tapply(df_all$deviceID, df_all$customerID, function(x) length(unique(x)))[as.character(df_all$customerID)]
# total number of types of payment per customer
df_all$payment_per_customer = tapply(as.character(df_all$paymentMethod), df_all$customerID, function(x) length(unique(x)))[as.character(df_all$customerID)]

# total original price per customer
df_all$original_price_per_customer = tapply(df_all$rrp*df_all$quantity, df_all$customerID, sum)[as.character(df_all$customerID)]
# total price per customer
df_all$price_per_customer = tapply(df_all$price, df_all$customerID, sum)[as.character(df_all$customerID)]
# total quantity of vouchers per customer
df_all$voucherID = as.character(df_all$voucherID)
df_all$voucherID[is.na(df_all$voucherID)] = "-1"
temp = aggregate(df_all$voucherID!="0", list(df_all$customerID, df_all$orderID), unique)
names(temp) = c("customerID", "orderID", "voucher_quantity_per_customer")
df_all$voucher_quantity_per_customer = tapply(temp$voucher_quantity_per_customer, temp$customerID, sum)[as.character(df_all$customerID)]
# total voucherAmount per customer
temp = aggregate(df_all$voucherAmount, list(df_all$customerID, df_all$orderID), unique)
names(temp) = c("customerID", "orderID", "voucherAmount_per_customer")
df_all$voucherAmount_per_customer = tapply(temp$voucherAmount_per_customer, temp$customerID, sum)[as.character(df_all$customerID)]
# total true price per customer
df_all$true_price_per_customer = df_all$price_per_customer - df_all$voucherAmount_per_customer
# total discounted price per customer
df_all$discounted_price_per_customer = df_all$original_price_per_customer - df_all$true_price_per_customer
# discounted ratio per customer
df_all$discounted_ratio_per_customer = df_all$discounted_price_per_customer/df_all$original_price_per_customer
df_all$discounted_ratio_per_customer[is.na(df_all$discounted_ratio_per_customer)] = 0
# mean, max, min rrp per customer
df_all$mean_rrp_per_customer = tapply(df_all$rrp, df_all$customerID, mean)[as.character(df_all$customerID)]
df_all$max_rrp_per_customer = tapply(df_all$rrp, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_rrp_per_customer = tapply(df_all$rrp, df_all$customerID, min)[as.character(df_all$customerID)]

# total quantity of choice items per customer
df_all$choice_item_quantity_per_customer = tapply(df_all$quantity*df_all$choice_item_index, df_all$customerID, sum)[as.character(df_all$customerID)]
# ratio of quantity of choice items per customer
df_all$choice_item_quantity_ratio_per_customer = df_all$choice_item_quantity_per_customer/df_all$quantity_per_customer
# total quantity of ac items per customer
df_all$ac_item_quantity_per_customer = tapply(df_all$quantity*df_all$ac_item_index, df_all$customerID, sum)[as.character(df_all$customerID)]
# ratio of quantity of ac items per customer
df_all$ac_item_quantity_ratio_per_customer = df_all$ac_item_quantity_per_customer/df_all$quantity_per_customer
# total quantity of as items per customer
df_all$as_item_quantity_per_customer = tapply(df_all$quantity*df_all$as_item_index, df_all$customerID, sum)[as.character(df_all$customerID)]
# ratio of quantity of as items per customer
df_all$as_item_quantity_ratio_per_customer = df_all$as_item_quantity_per_customer/df_all$quantity_per_customer
# total quantity of cp items per customer
df_all$cp_item_quantity_per_customer = tapply(df_all$quantity*df_all$cp_item_index, df_all$customerID, sum)[as.character(df_all$customerID)]
# ratio of quantity of cp items per customer
df_all$cp_item_quantity_ratio_per_customer = df_all$cp_item_quantity_per_customer/df_all$quantity_per_customer
# total quantity of sp items per customer
df_all$sp_item_quantity_per_customer = tapply(df_all$quantity*df_all$sp_item_index, df_all$customerID, sum)[as.character(df_all$customerID)]
# ratio of quantity of sp items per customer
df_all$sp_item_quantity_ratio_per_customer = df_all$sp_item_quantity_per_customer/df_all$quantity_per_customer
# total quantity of csp items per customer
df_all$csp_item_quantity_per_customer = tapply(df_all$quantity*df_all$csp_item_index, df_all$customerID, sum)[as.character(df_all$customerID)]
# ratio of quantity of csp items per customer
df_all$csp_item_quantity_ratio_per_customer = df_all$csp_item_quantity_per_customer/df_all$quantity_per_customer
# total quantity of pr items per customer
df_all$pr_item_quantity_per_customer = tapply(df_all$quantity*df_all$pr_item_index, df_all$customerID, sum)[as.character(df_all$customerID)]
# ratio of quantity of pr items per customer
df_all$pr_item_quantity_ratio_per_customer = df_all$pr_item_quantity_per_customer/df_all$quantity_per_customer
# total quantity of cpr items per customer
df_all$cpr_item_quantity_per_customer = tapply(df_all$quantity*df_all$cpr_item_index, df_all$customerID, sum)[as.character(df_all$customerID)]
# ratio of quantity of cpr items per customer
df_all$cpr_item_quantity_ratio_per_customer = df_all$cpr_item_quantity_per_customer/df_all$quantity_per_customer
# total quantity of spr items per customer
df_all$spr_item_quantity_per_customer = tapply(df_all$quantity*df_all$spr_item_index, df_all$customerID, sum)[as.character(df_all$customerID)]
# ratio of quantity of spr items per customer
df_all$spr_item_quantity_ratio_per_customer = df_all$spr_item_quantity_per_customer/df_all$quantity_per_customer
# total quantity of cspr items per customer
df_all$cspr_item_quantity_per_customer = tapply(df_all$quantity*df_all$cspr_item_index, df_all$customerID, sum)[as.character(df_all$customerID)]
# ratio of quantity of cspr items per customer
df_all$cspr_item_quantity_ratio_per_customer = df_all$cspr_item_quantity_per_customer/df_all$quantity_per_customer
# total quantity of choice_1 items per customer
df_all$choice_1_item_quantity_per_customer = tapply(df_all$quantity*df_all$choice_1_item_index, df_all$customerID, sum)[as.character(df_all$customerID)]
# ratio of quantity of choice_1 items per customer
df_all$choice_1_item_quantity_ratio_per_customer = df_all$choice_1_item_quantity_per_customer/df_all$quantity_per_customer
# total quantity of as_1 items per customer
df_all$as_1_item_quantity_per_customer = tapply(df_all$quantity*df_all$as_1_item_index, df_all$customerID, sum)[as.character(df_all$customerID)]
# ratio of quantity of as_1 items per customer
df_all$as_1_item_quantity_ratio_per_customer = df_all$as_1_item_quantity_per_customer/df_all$quantity_per_customer
# total quantity of sp_1 items per customer
df_all$sp_1_item_quantity_per_customer = tapply(df_all$quantity*df_all$sp_1_item_index, df_all$customerID, sum)[as.character(df_all$customerID)]
# ratio of quantity of sp_1 items per customer
df_all$sp_1_item_quantity_ratio_per_customer = df_all$sp_1_item_quantity_per_customer/df_all$quantity_per_customer
# total quantity of pr_1 items per customer
df_all$pr_1_item_quantity_per_customer = tapply(df_all$quantity*df_all$pr_1_item_index, df_all$customerID, sum)[as.character(df_all$customerID)]
# ratio of quantity of pr_1 items per customer
df_all$pr_1_item_quantity_ratio_per_customer = df_all$pr_1_item_quantity_per_customer/df_all$quantity_per_customer
# total quantity of spr_1 items per customer
df_all$spr_1_item_quantity_per_customer = tapply(df_all$quantity*df_all$spr_1_item_index, df_all$customerID, sum)[as.character(df_all$customerID)]
# ratio of quantity of spr_1 items per customer
df_all$spr_1_item_quantity_ratio_per_customer = df_all$spr_1_item_quantity_per_customer/df_all$quantity_per_customer
# total quantity of choice_234 items per customer
df_all$choice_234_item_quantity_per_customer = tapply(df_all$quantity*df_all$choice_234_item_index, df_all$customerID, sum)[as.character(df_all$customerID)]
# ratio of quantity of choice_234 items per customer
df_all$choice_234_item_quantity_ratio_per_customer = df_all$choice_234_item_quantity_per_customer/df_all$quantity_per_customer
# total quantity of as_234 items per customer
df_all$as_234_item_quantity_per_customer = tapply(df_all$quantity*df_all$as_234_item_index, df_all$customerID, sum)[as.character(df_all$customerID)]
# ratio of quantity of as_234 items per customer
df_all$as_234_item_quantity_ratio_per_customer = df_all$as_234_item_quantity_per_customer/df_all$quantity_per_customer
# total quantity of sp_234 items per customer
df_all$sp_234_item_quantity_per_customer = tapply(df_all$quantity*df_all$sp_234_item_index, df_all$customerID, sum)[as.character(df_all$customerID)]
# ratio of quantity of sp_234 items per customer
df_all$sp_234_item_quantity_ratio_per_customer = df_all$sp_234_item_quantity_per_customer/df_all$quantity_per_customer
# total quantity of pr_234 items per customer
df_all$pr_234_item_quantity_per_customer = tapply(df_all$quantity*df_all$pr_234_item_index, df_all$customerID, sum)[as.character(df_all$customerID)]
# ratio of quantity of pr_234 items per customer
df_all$pr_234_item_quantity_ratio_per_customer = df_all$pr_234_item_quantity_per_customer/df_all$quantity_per_customer
# total quantity of spr_234 items per customer
df_all$spr_234_item_quantity_per_customer = tapply(df_all$quantity*df_all$spr_234_item_index, df_all$customerID, sum)[as.character(df_all$customerID)]
# ratio of quantity of spr_234 items per customer
df_all$spr_234_item_quantity_ratio_per_customer = df_all$spr_234_item_quantity_per_customer/df_all$quantity_per_customer
# total quantity of choice_34 items per customer
df_all$choice_34_item_quantity_per_customer = tapply(df_all$quantity*df_all$choice_34_item_index, df_all$customerID, sum)[as.character(df_all$customerID)]
# ratio of quantity of choice_34 items per customer
df_all$choice_34_item_quantity_ratio_per_customer = df_all$choice_34_item_quantity_per_customer/df_all$quantity_per_customer
# total quantity of as_34 items per customer
df_all$as_34_item_quantity_per_customer = tapply(df_all$quantity*df_all$as_34_item_index, df_all$customerID, sum)[as.character(df_all$customerID)]
# ratio of quantity of as_34 items per customer
df_all$as_34_item_quantity_ratio_per_customer = df_all$as_34_item_quantity_per_customer/df_all$quantity_per_customer
# total quantity of sp_34 items per customer
df_all$sp_34_item_quantity_per_customer = tapply(df_all$quantity*df_all$sp_34_item_index, df_all$customerID, sum)[as.character(df_all$customerID)]
# ratio of quantity of sp_34 items per customer
df_all$sp_34_item_quantity_ratio_per_customer = df_all$sp_34_item_quantity_per_customer/df_all$quantity_per_customer
# total quantity of pr_34 items per customer
df_all$pr_34_item_quantity_per_customer = tapply(df_all$quantity*df_all$pr_34_item_index, df_all$customerID, sum)[as.character(df_all$customerID)]
# ratio of quantity of pr_34 items per customer
df_all$pr_34_item_quantity_ratio_per_customer = df_all$pr_34_item_quantity_per_customer/df_all$quantity_per_customer
# total quantity of spr_34 items per customer
df_all$spr_34_item_quantity_per_customer = tapply(df_all$quantity*df_all$spr_34_item_index, df_all$customerID, sum)[as.character(df_all$customerID)]
# ratio of quantity of spr_34 items per customer
df_all$spr_34_item_quantity_ratio_per_customer = df_all$spr_34_item_quantity_per_customer/df_all$quantity_per_customer
# total quantity of choice_4 items per customer
df_all$choice_4_item_quantity_per_customer = tapply(df_all$quantity*df_all$choice_4_item_index, df_all$customerID, sum)[as.character(df_all$customerID)]
# ratio of quantity of choice_4 items per customer
df_all$choice_4_item_quantity_ratio_per_customer = df_all$choice_4_item_quantity_per_customer/df_all$quantity_per_customer
# total quantity of as_4 items per customer
df_all$as_4_item_quantity_per_customer = tapply(df_all$quantity*df_all$as_4_item_index, df_all$customerID, sum)[as.character(df_all$customerID)]
# ratio of quantity of as_4 items per customer
df_all$as_4_item_quantity_ratio_per_customer = df_all$as_4_item_quantity_per_customer/df_all$quantity_per_customer
# total quantity of sp_4 items per customer
df_all$sp_4_item_quantity_per_customer = tapply(df_all$quantity*df_all$sp_4_item_index, df_all$customerID, sum)[as.character(df_all$customerID)]
# ratio of quantity of sp_4 items per customer
df_all$sp_4_item_quantity_ratio_per_customer = df_all$sp_4_item_quantity_per_customer/df_all$quantity_per_customer
# total quantity of pr_4 items per customer
df_all$pr_4_item_quantity_per_customer = tapply(df_all$quantity*df_all$pr_4_item_index, df_all$customerID, sum)[as.character(df_all$customerID)]
# ratio of quantity of pr_4 items per customer
df_all$pr_4_item_quantity_ratio_per_customer = df_all$pr_4_item_quantity_per_customer/df_all$quantity_per_customer
# total quantity of spr_4 items per customer
df_all$spr_4_item_quantity_per_customer = tapply(df_all$quantity*df_all$spr_4_item_index, df_all$customerID, sum)[as.character(df_all$customerID)]
# ratio of quantity of spr_4 items per customer
df_all$spr_4_item_quantity_ratio_per_customer = df_all$spr_4_item_quantity_per_customer/df_all$quantity_per_customer

# mean, max and min frequency of a single article per customer
temp = tapply(df_all$articleID, df_all$customerID, function(x) table(x))
df_all$mean_article_freq_per_customer = sapply(temp, mean)[as.character(df_all$customerID)]
df_all$max_article_freq_per_customer = sapply(temp, max)[as.character(df_all$customerID)]
df_all$min_article_freq_per_customer = sapply(temp, min)[as.character(df_all$customerID)]
# mean, max and min frequency of a single productGroup per customer
temp = tapply(df_all$productGroup, df_all$customerID, function(x) table(x))
df_all$mean_prod_freq_per_customer = sapply(temp, mean)[as.character(df_all$customerID)]
df_all$max_prod_freq_per_customer = sapply(temp, max)[as.character(df_all$customerID)]
df_all$min_prod_freq_per_customer = sapply(temp, min)[as.character(df_all$customerID)]
# mean, max and min frequency of a single colorCode per customer
temp = tapply(df_all$colorCode, df_all$customerID, function(x) table(x))
df_all$mean_color_freq_per_customer = sapply(temp, mean)[as.character(df_all$customerID)]
df_all$max_color_freq_per_customer = sapply(temp, max)[as.character(df_all$customerID)]
df_all$min_color_freq_per_customer = sapply(temp, min)[as.character(df_all$customerID)]
# mean, max and min frequency of a single sizeCode per customer
temp = tapply(as.character(df_all$sizeCode), df_all$customerID, function(x) table(x))
df_all$mean_size_freq_per_customer = sapply(temp, mean)[as.character(df_all$customerID)]
df_all$max_size_freq_per_customer = sapply(temp, max)[as.character(df_all$customerID)]
df_all$min_size_freq_per_customer = sapply(temp, min)[as.character(df_all$customerID)]

# customer-order feature ######################################################################################################
# mean, max and min frequency in a single order per customer
df_all$mean_order_freq_per_customer = df_all$freq_per_customer/df_all$order_per_customer
df_all$max_order_freq_per_customer = tapply(df_all$freq_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_freq_per_customer = tapply(df_all$freq_per_order, df_all$customerID, min)[as.character(df_all$customerID)]
# mean, max and min quantity in a single order per customer
df_all$mean_order_quantity_per_customer = df_all$quantity_per_customer/df_all$order_per_customer
df_all$max_order_quantity_per_customer = tapply(df_all$quantity_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_quantity_per_customer = tapply(df_all$quantity_per_order, df_all$customerID, min)[as.character(df_all$customerID)]
# mean, max and min number of types of articleID in a single order per customer
customer_list = tapply(df_all$customerID, df_all$orderID, unique)
df_all$mean_order_article_per_customer = tapply(tapply(df_all$articleID, df_all$orderID, function(x) length(unique(x))), customer_list, mean)[as.character(df_all$customerID)]
df_all$max_order_article_per_customer = tapply(df_all$article_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_article_per_customer = tapply(df_all$article_per_order, df_all$customerID, min)[as.character(df_all$customerID)]
# mean, max and min number of types of productGroup in a single order per customer
df_all$mean_order_prod_per_customer = tapply(tapply(df_all$productGroup, df_all$orderID, function(x) length(unique(x))), customer_list, mean)[as.character(df_all$customerID)]
df_all$max_order_prod_per_customer = tapply(df_all$prod_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_prod_per_customer = tapply(df_all$prod_per_order, df_all$customerID, min)[as.character(df_all$customerID)]
# mean, max and min number of types of colorCode in a single order per customer
df_all$mean_order_color_per_customer = tapply(df_all$color_per_order, df_all$customerID, mean)[as.character(df_all$customerID)]
df_all$max_order_color_per_customer = tapply(df_all$color_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_color_per_customer = tapply(df_all$color_per_order, df_all$customerID, min)[as.character(df_all$customerID)]
# mean, max and min number of types of sizeCode in a single order per customer
df_all$mean_order_size_per_customer = tapply(df_all$size_per_order, df_all$customerID, mean)[as.character(df_all$customerID)]
df_all$max_order_size_per_customer = tapply(df_all$size_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_size_per_customer = tapply(df_all$size_per_order, df_all$customerID, min)[as.character(df_all$customerID)]

# mean, max and min original price of a single order per customer
df_all$mean_order_original_price_per_customer = df_all$original_price_per_customer/df_all$order_per_customer
df_all$max_order_original_price_per_customer = tapply(df_all$original_price_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_original_price_per_customer = tapply(df_all$original_price_per_order, df_all$customerID, min)[as.character(df_all$customerID)]
# mean, max and min price of a single order per customer
df_all$mean_order_price_per_customer = df_all$price_per_customer/df_all$order_per_customer
df_all$max_order_price_per_customer = tapply(df_all$price_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_price_per_customer = tapply(df_all$price_per_order, df_all$customerID, min)[as.character(df_all$customerID)]
# mean, max and min voucherAmount of a single order per customer
df_all$mean_order_voucherAmount_per_customer = df_all$voucherAmount_per_customer/df_all$order_per_customer
df_all$max_order_voucherAmount_per_customer = tapply(df_all$voucherAmount, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_voucherAmount_per_customer = tapply(df_all$voucherAmount, df_all$customerID, min)[as.character(df_all$customerID)]
# mean, max and min true price of a single order per customer
df_all$mean_order_true_price_per_customer = df_all$true_price_per_customer/df_all$order_per_customer
df_all$max_order_true_price_per_customer = tapply(df_all$true_price_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_true_price_per_customer = tapply(df_all$true_price_per_order, df_all$customerID, min)[as.character(df_all$customerID)]
# mean, max and min discounted price of a single order per customer
df_all$mean_order_discounted_price_per_customer = df_all$discounted_price_per_customer/df_all$order_per_customer
df_all$max_order_discounted_price_per_customer = tapply(df_all$discounted_price_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_discounted_price_per_customer = tapply(df_all$discounted_price_per_order, df_all$customerID, min)[as.character(df_all$customerID)]
# max and min discounted ratio of a single order per customer
df_all$max_order_discounted_ratio_per_customer = tapply(df_all$discounted_ratio_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_discounted_ratio_per_customer = tapply(df_all$discounted_ratio_per_order, df_all$customerID, min)[as.character(df_all$customerID)]

# mean, max and min number of types of choice items in a single order per customer
df_all$mean_order_choice_item_per_customer = tapply((tapply(df_all$articleID, df_all$orderID, function(x) length(unique(x)))-tapply(1-df_all$choice_item_index, df_all$orderID, sum)), customer_list, mean)[as.character(df_all$customerID)]
df_all$max_order_choice_item_per_customer = tapply(df_all$choice_item_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_choice_item_per_customer = tapply(df_all$choice_item_per_order, df_all$customerID, min)[as.character(df_all$customerID)]
# max and min ratio of types of choice items in a single order per customer
df_all$max_order_choice_item_ratio_per_customer = tapply(df_all$choice_item_ratio_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_choice_item_ratio_per_customer = tapply(df_all$choice_item_ratio_per_order, df_all$customerID, min)[as.character(df_all$customerID)]
# mean, max and min quantity of choice items in a single order per customer
df_all$mean_order_choice_item_quantity_per_customer = df_all$choice_item_quantity_per_customer/df_all$order_per_customer
df_all$max_order_choice_item_quantity_per_customer = tapply(df_all$choice_item_quantity_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_choice_item_quantity_per_customer = tapply(df_all$choice_item_quantity_per_order, df_all$customerID, min)[as.character(df_all$customerID)]
# max and min quantity ratio of choice items in a single order per customer
df_all$max_order_choice_item_quantity_ratio_per_customer = tapply(df_all$choice_item_quantity_ratio_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_choice_item_quantity_ratio_per_customer = tapply(df_all$choice_item_quantity_ratio_per_order, df_all$customerID, min)[as.character(df_all$customerID)]

# mean, max and min quantity of ac items in a single order per customer
df_all$mean_order_ac_item_quantity_per_customer = df_all$ac_item_quantity_per_customer/df_all$order_per_customer
df_all$max_order_ac_item_quantity_per_customer = tapply(df_all$ac_item_quantity_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_ac_item_quantity_per_customer = tapply(df_all$ac_item_quantity_per_order, df_all$customerID, min)[as.character(df_all$customerID)]
# max and min quantity ratio of ac items in a single order per customer
df_all$max_order_ac_item_quantity_ratio_per_customer = tapply(df_all$ac_item_quantity_ratio_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_ac_item_quantity_ratio_per_customer = tapply(df_all$ac_item_quantity_ratio_per_order, df_all$customerID, min)[as.character(df_all$customerID)]

# mean, max and min quantity of as items in a single order per customer
df_all$mean_order_as_item_quantity_per_customer = df_all$as_item_quantity_per_customer/df_all$order_per_customer
df_all$max_order_as_item_quantity_per_customer = tapply(df_all$as_item_quantity_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_as_item_quantity_per_customer = tapply(df_all$as_item_quantity_per_order, df_all$customerID, min)[as.character(df_all$customerID)]
# max and min quantity ratio of as items in a single order per customer
df_all$max_order_as_item_quantity_ratio_per_customer = tapply(df_all$as_item_quantity_ratio_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_as_item_quantity_ratio_per_customer = tapply(df_all$as_item_quantity_ratio_per_order, df_all$customerID, min)[as.character(df_all$customerID)]

# mean, max and min quantity of cp items in a single order per customer
df_all$mean_order_cp_item_quantity_per_customer = df_all$cp_item_quantity_per_customer/df_all$order_per_customer
df_all$max_order_cp_item_quantity_per_customer = tapply(df_all$cp_item_quantity_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_cp_item_quantity_per_customer = tapply(df_all$cp_item_quantity_per_order, df_all$customerID, min)[as.character(df_all$customerID)]
# max and min quantity ratio of cp items in a single order per customer
df_all$max_order_cp_item_quantity_ratio_per_customer = tapply(df_all$cp_item_quantity_ratio_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_cp_item_quantity_ratio_per_customer = tapply(df_all$cp_item_quantity_ratio_per_order, df_all$customerID, min)[as.character(df_all$customerID)]

# mean, max and min quantity of sp items in a single order per customer
df_all$mean_order_sp_item_quantity_per_customer = df_all$sp_item_quantity_per_customer/df_all$order_per_customer
df_all$max_order_sp_item_quantity_per_customer = tapply(df_all$sp_item_quantity_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_sp_item_quantity_per_customer = tapply(df_all$sp_item_quantity_per_order, df_all$customerID, min)[as.character(df_all$customerID)]
# max and min quantity ratio of sp items in a single order per customer
df_all$max_order_sp_item_quantity_ratio_per_customer = tapply(df_all$sp_item_quantity_ratio_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_sp_item_quantity_ratio_per_customer = tapply(df_all$sp_item_quantity_ratio_per_order, df_all$customerID, min)[as.character(df_all$customerID)]

# mean, max and min quantity of csp items in a single order per customer
df_all$mean_order_csp_item_quantity_per_customer = df_all$csp_item_quantity_per_customer/df_all$order_per_customer
df_all$max_order_csp_item_quantity_per_customer = tapply(df_all$csp_item_quantity_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_csp_item_quantity_per_customer = tapply(df_all$csp_item_quantity_per_order, df_all$customerID, min)[as.character(df_all$customerID)]
# max and min quantity ratio of csp items in a single order per customer
df_all$max_order_csp_item_quantity_ratio_per_customer = tapply(df_all$csp_item_quantity_ratio_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_csp_item_quantity_ratio_per_customer = tapply(df_all$csp_item_quantity_ratio_per_order, df_all$customerID, min)[as.character(df_all$customerID)]

# mean, max and min quantity of pr items in a single order per customer
df_all$mean_order_pr_item_quantity_per_customer = df_all$pr_item_quantity_per_customer/df_all$order_per_customer
df_all$max_order_pr_item_quantity_per_customer = tapply(df_all$pr_item_quantity_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_pr_item_quantity_per_customer = tapply(df_all$pr_item_quantity_per_order, df_all$customerID, min)[as.character(df_all$customerID)]
# max and min quantity ratio of pr items in a single order per customer
df_all$max_order_pr_item_quantity_ratio_per_customer = tapply(df_all$pr_item_quantity_ratio_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_pr_item_quantity_ratio_per_customer = tapply(df_all$pr_item_quantity_ratio_per_order, df_all$customerID, min)[as.character(df_all$customerID)]

# mean, max and min quantity of cpr items in a single order per customer
df_all$mean_order_cpr_item_quantity_per_customer = df_all$cpr_item_quantity_per_customer/df_all$order_per_customer
df_all$max_order_cpr_item_quantity_per_customer = tapply(df_all$cpr_item_quantity_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_cpr_item_quantity_per_customer = tapply(df_all$cpr_item_quantity_per_order, df_all$customerID, min)[as.character(df_all$customerID)]
# max and min quantity ratio of cpr items in a single order per customer
df_all$max_order_cpr_item_quantity_ratio_per_customer = tapply(df_all$cpr_item_quantity_ratio_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_cpr_item_quantity_ratio_per_customer = tapply(df_all$cpr_item_quantity_ratio_per_order, df_all$customerID, min)[as.character(df_all$customerID)]

# mean, max and min quantity of spr items in a single order per customer
df_all$mean_order_spr_item_quantity_per_customer = df_all$spr_item_quantity_per_customer/df_all$order_per_customer
df_all$max_order_spr_item_quantity_per_customer = tapply(df_all$spr_item_quantity_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_spr_item_quantity_per_customer = tapply(df_all$spr_item_quantity_per_order, df_all$customerID, min)[as.character(df_all$customerID)]
# max and min quantity ratio of spr items in a single order per customer
df_all$max_order_spr_item_quantity_ratio_per_customer = tapply(df_all$spr_item_quantity_ratio_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_spr_item_quantity_ratio_per_customer = tapply(df_all$spr_item_quantity_ratio_per_order, df_all$customerID, min)[as.character(df_all$customerID)]

# mean, max and min quantity of cspr items in a single order per customer
df_all$mean_order_cspr_item_quantity_per_customer = df_all$cspr_item_quantity_per_customer/df_all$order_per_customer
df_all$max_order_cspr_item_quantity_per_customer = tapply(df_all$cspr_item_quantity_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_cspr_item_quantity_per_customer = tapply(df_all$cspr_item_quantity_per_order, df_all$customerID, min)[as.character(df_all$customerID)]
# max and min quantity ratio of cspr items in a single order per customer
df_all$max_order_cspr_item_quantity_ratio_per_customer = tapply(df_all$cspr_item_quantity_ratio_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_cspr_item_quantity_ratio_per_customer = tapply(df_all$cspr_item_quantity_ratio_per_order, df_all$customerID, min)[as.character(df_all$customerID)]

# mean, max and min quantity of choice_1 items in a single order per customer
df_all$mean_order_choice_1_item_quantity_per_customer = df_all$choice_1_item_quantity_per_customer/df_all$order_per_customer
df_all$max_order_choice_1_item_quantity_per_customer = tapply(df_all$choice_1_item_quantity_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_choice_1_item_quantity_per_customer = tapply(df_all$choice_1_item_quantity_per_order, df_all$customerID, min)[as.character(df_all$customerID)]
# max and min quantity ratio of choice_1 items in a single order per customer
df_all$max_order_choice_1_item_quantity_ratio_per_customer = tapply(df_all$choice_1_item_quantity_ratio_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_choice_1_item_quantity_ratio_per_customer = tapply(df_all$choice_1_item_quantity_ratio_per_order, df_all$customerID, min)[as.character(df_all$customerID)]
# mean, max and min quantity of as_1 items in a single order per customer
df_all$mean_order_as_1_item_quantity_per_customer = df_all$as_1_item_quantity_per_customer/df_all$order_per_customer
df_all$max_order_as_1_item_quantity_per_customer = tapply(df_all$as_1_item_quantity_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_as_1_item_quantity_per_customer = tapply(df_all$as_1_item_quantity_per_order, df_all$customerID, min)[as.character(df_all$customerID)]
# max and min quantity ratio of as_1 items in a single order per customer
df_all$max_order_as_1_item_quantity_ratio_per_customer = tapply(df_all$as_1_item_quantity_ratio_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_as_1_item_quantity_ratio_per_customer = tapply(df_all$as_1_item_quantity_ratio_per_order, df_all$customerID, min)[as.character(df_all$customerID)]
# mean, max and min quantity of sp_1 items in a single order per customer
df_all$mean_order_sp_1_item_quantity_per_customer = df_all$sp_1_item_quantity_per_customer/df_all$order_per_customer
df_all$max_order_sp_1_item_quantity_per_customer = tapply(df_all$sp_1_item_quantity_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_sp_1_item_quantity_per_customer = tapply(df_all$sp_1_item_quantity_per_order, df_all$customerID, min)[as.character(df_all$customerID)]
# max and min quantity ratio of sp_1 items in a single order per customer
df_all$max_order_sp_1_item_quantity_ratio_per_customer = tapply(df_all$sp_1_item_quantity_ratio_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_sp_1_item_quantity_ratio_per_customer = tapply(df_all$sp_1_item_quantity_ratio_per_order, df_all$customerID, min)[as.character(df_all$customerID)]
# mean, max and min quantity of pr_1 items in a single order per customer
df_all$mean_order_pr_1_item_quantity_per_customer = df_all$pr_1_item_quantity_per_customer/df_all$order_per_customer
df_all$max_order_pr_1_item_quantity_per_customer = tapply(df_all$pr_1_item_quantity_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_pr_1_item_quantity_per_customer = tapply(df_all$pr_1_item_quantity_per_order, df_all$customerID, min)[as.character(df_all$customerID)]
# max and min quantity ratio of pr_1 items in a single order per customer
df_all$max_order_pr_1_item_quantity_ratio_per_customer = tapply(df_all$pr_1_item_quantity_ratio_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_pr_1_item_quantity_ratio_per_customer = tapply(df_all$pr_1_item_quantity_ratio_per_order, df_all$customerID, min)[as.character(df_all$customerID)]
# mean, max and min quantity of spr_1 items in a single order per customer
df_all$mean_order_spr_1_item_quantity_per_customer = df_all$spr_1_item_quantity_per_customer/df_all$order_per_customer
df_all$max_order_spr_1_item_quantity_per_customer = tapply(df_all$spr_1_item_quantity_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_spr_1_item_quantity_per_customer = tapply(df_all$spr_1_item_quantity_per_order, df_all$customerID, min)[as.character(df_all$customerID)]
# max and min quantity ratio of spr_1 items in a single order per customer
df_all$max_order_spr_1_item_quantity_ratio_per_customer = tapply(df_all$spr_1_item_quantity_ratio_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_spr_1_item_quantity_ratio_per_customer = tapply(df_all$spr_1_item_quantity_ratio_per_order, df_all$customerID, min)[as.character(df_all$customerID)]

# mean, max and min quantity of choice_234 items in a single order per customer
df_all$mean_order_choice_234_item_quantity_per_customer = df_all$choice_234_item_quantity_per_customer/df_all$order_per_customer
df_all$max_order_choice_234_item_quantity_per_customer = tapply(df_all$choice_234_item_quantity_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_choice_234_item_quantity_per_customer = tapply(df_all$choice_234_item_quantity_per_order, df_all$customerID, min)[as.character(df_all$customerID)]
# max and min quantity ratio of choice_234 items in a single order per customer
df_all$max_order_choice_234_item_quantity_ratio_per_customer = tapply(df_all$choice_234_item_quantity_ratio_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_choice_234_item_quantity_ratio_per_customer = tapply(df_all$choice_234_item_quantity_ratio_per_order, df_all$customerID, min)[as.character(df_all$customerID)]
# mean, max and min quantity of as_234 items in a single order per customer
df_all$mean_order_as_234_item_quantity_per_customer = df_all$as_234_item_quantity_per_customer/df_all$order_per_customer
df_all$max_order_as_234_item_quantity_per_customer = tapply(df_all$as_234_item_quantity_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_as_234_item_quantity_per_customer = tapply(df_all$as_234_item_quantity_per_order, df_all$customerID, min)[as.character(df_all$customerID)]
# max and min quantity ratio of as_234 items in a single order per customer
df_all$max_order_as_234_item_quantity_ratio_per_customer = tapply(df_all$as_234_item_quantity_ratio_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_as_234_item_quantity_ratio_per_customer = tapply(df_all$as_234_item_quantity_ratio_per_order, df_all$customerID, min)[as.character(df_all$customerID)]
# mean, max and min quantity of sp_234 items in a single order per customer
df_all$mean_order_sp_234_item_quantity_per_customer = df_all$sp_234_item_quantity_per_customer/df_all$order_per_customer
df_all$max_order_sp_234_item_quantity_per_customer = tapply(df_all$sp_234_item_quantity_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_sp_234_item_quantity_per_customer = tapply(df_all$sp_234_item_quantity_per_order, df_all$customerID, min)[as.character(df_all$customerID)]
# max and min quantity ratio of sp_234 items in a single order per customer
df_all$max_order_sp_234_item_quantity_ratio_per_customer = tapply(df_all$sp_234_item_quantity_ratio_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_sp_234_item_quantity_ratio_per_customer = tapply(df_all$sp_234_item_quantity_ratio_per_order, df_all$customerID, min)[as.character(df_all$customerID)]
# mean, max and min quantity of pr_234 items in a single order per customer
df_all$mean_order_pr_234_item_quantity_per_customer = df_all$pr_234_item_quantity_per_customer/df_all$order_per_customer
df_all$max_order_pr_234_item_quantity_per_customer = tapply(df_all$pr_234_item_quantity_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_pr_234_item_quantity_per_customer = tapply(df_all$pr_234_item_quantity_per_order, df_all$customerID, min)[as.character(df_all$customerID)]
# max and min quantity ratio of pr_234 items in a single order per customer
df_all$max_order_pr_234_item_quantity_ratio_per_customer = tapply(df_all$pr_234_item_quantity_ratio_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_pr_234_item_quantity_ratio_per_customer = tapply(df_all$pr_234_item_quantity_ratio_per_order, df_all$customerID, min)[as.character(df_all$customerID)]
# mean, max and min quantity of spr_234 items in a single order per customer
df_all$mean_order_spr_234_item_quantity_per_customer = df_all$spr_234_item_quantity_per_customer/df_all$order_per_customer
df_all$max_order_spr_234_item_quantity_per_customer = tapply(df_all$spr_234_item_quantity_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_spr_234_item_quantity_per_customer = tapply(df_all$spr_234_item_quantity_per_order, df_all$customerID, min)[as.character(df_all$customerID)]
# max and min quantity ratio of spr_234 items in a single order per customer
df_all$max_order_spr_234_item_quantity_ratio_per_customer = tapply(df_all$spr_234_item_quantity_ratio_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_spr_234_item_quantity_ratio_per_customer = tapply(df_all$spr_234_item_quantity_ratio_per_order, df_all$customerID, min)[as.character(df_all$customerID)]

# mean, max and min quantity of choice_34 items in a single order per customer
df_all$mean_order_choice_34_item_quantity_per_customer = df_all$choice_34_item_quantity_per_customer/df_all$order_per_customer
df_all$max_order_choice_34_item_quantity_per_customer = tapply(df_all$choice_34_item_quantity_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_choice_34_item_quantity_per_customer = tapply(df_all$choice_34_item_quantity_per_order, df_all$customerID, min)[as.character(df_all$customerID)]
# max and min quantity ratio of choice_34 items in a single order per customer
df_all$max_order_choice_34_item_quantity_ratio_per_customer = tapply(df_all$choice_34_item_quantity_ratio_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_choice_34_item_quantity_ratio_per_customer = tapply(df_all$choice_34_item_quantity_ratio_per_order, df_all$customerID, min)[as.character(df_all$customerID)]
# mean, max and min quantity of as_34 items in a single order per customer
df_all$mean_order_as_34_item_quantity_per_customer = df_all$as_34_item_quantity_per_customer/df_all$order_per_customer
df_all$max_order_as_34_item_quantity_per_customer = tapply(df_all$as_34_item_quantity_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_as_34_item_quantity_per_customer = tapply(df_all$as_34_item_quantity_per_order, df_all$customerID, min)[as.character(df_all$customerID)]
# max and min quantity ratio of as_34 items in a single order per customer
df_all$max_order_as_34_item_quantity_ratio_per_customer = tapply(df_all$as_34_item_quantity_ratio_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_as_34_item_quantity_ratio_per_customer = tapply(df_all$as_34_item_quantity_ratio_per_order, df_all$customerID, min)[as.character(df_all$customerID)]
# mean, max and min quantity of sp_34 items in a single order per customer
df_all$mean_order_sp_34_item_quantity_per_customer = df_all$sp_34_item_quantity_per_customer/df_all$order_per_customer
df_all$max_order_sp_34_item_quantity_per_customer = tapply(df_all$sp_34_item_quantity_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_sp_34_item_quantity_per_customer = tapply(df_all$sp_34_item_quantity_per_order, df_all$customerID, min)[as.character(df_all$customerID)]
# max and min quantity ratio of sp_34 items in a single order per customer
df_all$max_order_sp_34_item_quantity_ratio_per_customer = tapply(df_all$sp_34_item_quantity_ratio_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_sp_34_item_quantity_ratio_per_customer = tapply(df_all$sp_34_item_quantity_ratio_per_order, df_all$customerID, min)[as.character(df_all$customerID)]
# mean, max and min quantity of pr_34 items in a single order per customer
df_all$mean_order_pr_34_item_quantity_per_customer = df_all$pr_34_item_quantity_per_customer/df_all$order_per_customer
df_all$max_order_pr_34_item_quantity_per_customer = tapply(df_all$pr_34_item_quantity_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_pr_34_item_quantity_per_customer = tapply(df_all$pr_34_item_quantity_per_order, df_all$customerID, min)[as.character(df_all$customerID)]
# max and min quantity ratio of pr_34 items in a single order per customer
df_all$max_order_pr_34_item_quantity_ratio_per_customer = tapply(df_all$pr_34_item_quantity_ratio_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_pr_34_item_quantity_ratio_per_customer = tapply(df_all$pr_34_item_quantity_ratio_per_order, df_all$customerID, min)[as.character(df_all$customerID)]
# mean, max and min quantity of spr_34 items in a single order per customer
df_all$mean_order_spr_34_item_quantity_per_customer = df_all$spr_34_item_quantity_per_customer/df_all$order_per_customer
df_all$max_order_spr_34_item_quantity_per_customer = tapply(df_all$spr_34_item_quantity_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_spr_34_item_quantity_per_customer = tapply(df_all$spr_34_item_quantity_per_order, df_all$customerID, min)[as.character(df_all$customerID)]
# max and min quantity ratio of spr_34 items in a single order per customer
df_all$max_order_spr_34_item_quantity_ratio_per_customer = tapply(df_all$spr_34_item_quantity_ratio_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_spr_34_item_quantity_ratio_per_customer = tapply(df_all$spr_34_item_quantity_ratio_per_order, df_all$customerID, min)[as.character(df_all$customerID)]

# mean, max and min quantity of choice_4 items in a single order per customer
df_all$mean_order_choice_4_item_quantity_per_customer = df_all$choice_4_item_quantity_per_customer/df_all$order_per_customer
df_all$max_order_choice_4_item_quantity_per_customer = tapply(df_all$choice_4_item_quantity_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_choice_4_item_quantity_per_customer = tapply(df_all$choice_4_item_quantity_per_order, df_all$customerID, min)[as.character(df_all$customerID)]
# max and min quantity ratio of choice_4 items in a single order per customer
df_all$max_order_choice_4_item_quantity_ratio_per_customer = tapply(df_all$choice_4_item_quantity_ratio_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_choice_4_item_quantity_ratio_per_customer = tapply(df_all$choice_4_item_quantity_ratio_per_order, df_all$customerID, min)[as.character(df_all$customerID)]
# mean, max and min quantity of as_4 items in a single order per customer
df_all$mean_order_as_4_item_quantity_per_customer = df_all$as_4_item_quantity_per_customer/df_all$order_per_customer
df_all$max_order_as_4_item_quantity_per_customer = tapply(df_all$as_4_item_quantity_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_as_4_item_quantity_per_customer = tapply(df_all$as_4_item_quantity_per_order, df_all$customerID, min)[as.character(df_all$customerID)]
# max and min quantity ratio of as_4 items in a single order per customer
df_all$max_order_as_4_item_quantity_ratio_per_customer = tapply(df_all$as_4_item_quantity_ratio_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_as_4_item_quantity_ratio_per_customer = tapply(df_all$as_4_item_quantity_ratio_per_order, df_all$customerID, min)[as.character(df_all$customerID)]
# mean, max and min quantity of sp_4 items in a single order per customer
df_all$mean_order_sp_4_item_quantity_per_customer = df_all$sp_4_item_quantity_per_customer/df_all$order_per_customer
df_all$max_order_sp_4_item_quantity_per_customer = tapply(df_all$sp_4_item_quantity_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_sp_4_item_quantity_per_customer = tapply(df_all$sp_4_item_quantity_per_order, df_all$customerID, min)[as.character(df_all$customerID)]
# max and min quantity ratio of sp_4 items in a single order per customer
df_all$max_order_sp_4_item_quantity_ratio_per_customer = tapply(df_all$sp_4_item_quantity_ratio_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_sp_4_item_quantity_ratio_per_customer = tapply(df_all$sp_4_item_quantity_ratio_per_order, df_all$customerID, min)[as.character(df_all$customerID)]
# mean, max and min quantity of pr_4 items in a single order per customer
df_all$mean_order_pr_4_item_quantity_per_customer = df_all$pr_4_item_quantity_per_customer/df_all$order_per_customer
df_all$max_order_pr_4_item_quantity_per_customer = tapply(df_all$pr_4_item_quantity_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_pr_4_item_quantity_per_customer = tapply(df_all$pr_4_item_quantity_per_order, df_all$customerID, min)[as.character(df_all$customerID)]
# max and min quantity ratio of pr_4 items in a single order per customer
df_all$max_order_pr_4_item_quantity_ratio_per_customer = tapply(df_all$pr_4_item_quantity_ratio_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_pr_4_item_quantity_ratio_per_customer = tapply(df_all$pr_4_item_quantity_ratio_per_order, df_all$customerID, min)[as.character(df_all$customerID)]
# mean, max and min quantity of spr_4 items in a single order per customer
df_all$mean_order_spr_4_item_quantity_per_customer = df_all$spr_4_item_quantity_per_customer/df_all$order_per_customer
df_all$max_order_spr_4_item_quantity_per_customer = tapply(df_all$spr_4_item_quantity_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_spr_4_item_quantity_per_customer = tapply(df_all$spr_4_item_quantity_per_order, df_all$customerID, min)[as.character(df_all$customerID)]
# max and min quantity ratio of spr_4 items in a single order per customer
df_all$max_order_spr_4_item_quantity_ratio_per_customer = tapply(df_all$spr_4_item_quantity_ratio_per_order, df_all$customerID, max)[as.character(df_all$customerID)]
df_all$min_order_spr_4_item_quantity_ratio_per_customer = tapply(df_all$spr_4_item_quantity_ratio_per_order, df_all$customerID, min)[as.character(df_all$customerID)]

# article feature #############################################################################################################
# total frequency of order items per article
df_all$freq_per_article = tapply(df_all$quantity, df_all$articleID, length)[as.character(df_all$articleID)]
# total quantity per article
df_all$quantity_per_article = tapply(df_all$quantity, df_all$articleID, sum)[as.character(df_all$articleID)]
# total number of orders per article
df_all$order_per_article = tapply(as.character(df_all$orderID), df_all$articleID, function(x) length(unique(x)))[as.character(df_all$articleID)]
# total number of customers per article
df_all$customer_per_article = tapply(df_all$customerID, df_all$articleID, function(x) length(unique(x)))[as.character(df_all$articleID)]

# mean, max and min of rrp per article
df_all$mean_rrp_per_article = tapply(df_all$rrp, df_all$articleID, mean)[as.character(df_all$articleID)]
df_all$max_rrp_per_article = tapply(df_all$rrp, df_all$articleID, max)[as.character(df_all$articleID)]
df_all$min_rrp_per_article = tapply(df_all$rrp, df_all$articleID, min)[as.character(df_all$articleID)]
# mean, max and min of discounted price per article
df_all$mean_discounted_price_per_article = tapply(df_all$discounted_price_per_quantity, df_all$articleID, mean)[as.character(df_all$articleID)]
df_all$max_discounted_price_per_article = tapply(df_all$discounted_price_per_quantity, df_all$articleID, max)[as.character(df_all$articleID)]
df_all$min_discounted_price_per_article = tapply(df_all$discounted_price_per_quantity, df_all$articleID, min)[as.character(df_all$articleID)]
# mean, max and min of discounted ratio per article
df_all$mean_discounted_ratio_per_article = tapply(df_all$discounted_ratio_per_quantity, df_all$articleID, mean)[as.character(df_all$articleID)]
df_all$max_discounted_ratio_per_article = tapply(df_all$discounted_ratio_per_quantity, df_all$articleID, max)[as.character(df_all$articleID)]
df_all$min_discounted_ratio_per_article = tapply(df_all$discounted_ratio_per_quantity, df_all$articleID, min)[as.character(df_all$articleID)]

# orderDate feature #############################################################################################################
df_all$first_orderDate = tapply(as.character(df_all$orderDate), df_all$customerID, min)[as.character(df_all$customerID)]
first_order_ymd = as.data.frame(str_split_fixed(df_all$first_orderDate, '-', 3))
df_all$first_order_year = as.numeric(first_order_ymd[, 1])
df_all$first_order_month = as.numeric(first_order_ymd[, 2])
df_all$first_order_day = as.numeric(first_order_ymd[, 3])

# payment feature ###############################################################################################################
payment_new = as.character(df_all$paymentMethod)
payment_new[payment_new %in% c("BPRG", "KGRG", "RG", "BPPL")] = "INVOICE"
payment_new[payment_new %in% c("PAYPALVC", "CBA", "BPLS", "KKE")] = "RIGHTAWAY"
df_all$invoice_ind = payment_new=="INVOICE"
df_all$rightaway_ind = payment_new=="RIGHTAWAY"

# two way table ###############################################################################################################
# quantity per customerID and articleID
temp = aggregate(df_all$quantity, list(df_all$customerID, df_all$articleID), sum)
names(temp) = c("customerID", "articleID", "quantity_per_customer_article")
df_all = join(df_all, temp, by=c("customerID", "articleID"))
# quantity per customerID and productGroup
temp = aggregate(df_all$quantity, list(df_all$customerID, df_all$productGroup), sum)
names(temp) = c("customerID", "productGroup", "quantity_per_customer_prod")
df_all = join(df_all, temp, by=c("customerID", "productGroup"))
# quantity per customerID, productGroup and rrp
temp = aggregate(df_all$quantity, list(df_all$customerID, df_all$productGroup, df_all$rrp), sum)
names(temp) = c("customerID", "productGroup", "rrp", "quantity_per_customer_prod_rrp")
df_all = join(df_all, temp, by=c("customerID", "productGroup", "rrp"))
# quantity per customerID and colorCode
temp = aggregate(df_all$quantity, list(df_all$customerID, df_all$colorCode), sum)
names(temp) = c("customerID", "colorCode", "quantity_per_customer_color")
df_all = join(df_all, temp, by=c("customerID", "colorCode"))
# quantity per customerID and sizeCode
temp = aggregate(df_all$quantity, list(df_all$customerID, df_all$sizeCode), sum)
names(temp) = c("customerID", "sizeCode", "quantity_per_customer_size")
df_all = join(df_all, temp, by=c("customerID", "sizeCode"))

# price per customerID and articleID
temp = aggregate(df_all$price, list(df_all$customerID, df_all$articleID), sum)
names(temp) = c("customerID", "articleID", "price_per_customer_article")
df_all = join(df_all, temp, by=c("customerID", "articleID"))
# price per customerID and productGroup
temp = aggregate(df_all$price, list(df_all$customerID, df_all$productGroup), sum)
names(temp) = c("customerID", "productGroup", "price_per_customer_prod")
df_all = join(df_all, temp, by=c("customerID", "productGroup"))
# price per customerID, productGroup and rrp
temp = aggregate(df_all$price, list(df_all$customerID, df_all$productGroup, df_all$rrp), sum)
names(temp) = c("customerID", "productGroup", "rrp", "price_per_customer_prod_rrp")
df_all = join(df_all, temp, by=c("customerID", "productGroup", "rrp"))
# price per customerID and colorCode
temp = aggregate(df_all$price, list(df_all$customerID, df_all$colorCode), sum)
names(temp) = c("customerID", "colorCode", "price_per_customer_color")
df_all = join(df_all, temp, by=c("customerID", "colorCode"))
# price per customerID and sizeCode
temp = aggregate(df_all$price, list(df_all$customerID, df_all$sizeCode), sum)
names(temp) = c("customerID", "sizeCode", "price_per_customer_size")
df_all = join(df_all, temp, by=c("customerID", "sizeCode"))

# discounted price per customerID and articleID
temp = aggregate(df_all$discounted_price_per_quantity*df_all$quantity, list(df_all$customerID, df_all$articleID), sum)
names(temp) = c("customerID", "articleID", "discounted_price_per_customer_article")
df_all = join(df_all, temp, by=c("customerID", "articleID"))
# discounted price per customerID and productGroup
temp = aggregate(df_all$discounted_price_per_quantity*df_all$quantity, list(df_all$customerID, df_all$productGroup), sum)
names(temp) = c("customerID", "productGroup", "discounted_price_per_customer_prod")
df_all = join(df_all, temp, by=c("customerID", "productGroup"))
# discounted price per customerID, productGroup and rrp
temp = aggregate(df_all$discounted_price_per_quantity*df_all$quantity, list(df_all$customerID, df_all$productGroup, df_all$rrp), sum)
names(temp) = c("customerID", "productGroup", "rrp", "discounted_price_per_customer_prod_rrp")
df_all = join(df_all, temp, by=c("customerID", "productGroup", "rrp"))
# discounted price per customerID and colorCode
temp = aggregate(df_all$discounted_price_per_quantity*df_all$quantity, list(df_all$customerID, df_all$colorCode), sum)
names(temp) = c("customerID", "colorCode", "discounted_price_per_customer_color")
df_all = join(df_all, temp, by=c("customerID", "colorCode"))
# discounted price per customerID and sizeCode
temp = aggregate(df_all$discounted_price_per_quantity*df_all$quantity, list(df_all$customerID, df_all$sizeCode), sum)
names(temp) = c("customerID", "sizeCode", "discounted_price_per_customer_size")
df_all = join(df_all, temp, by=c("customerID", "sizeCode"))

# quantity per orderID and article ID
temp = aggregate(df_all$quantity, list(df_all$orderID, df_all$articleID), sum)
names(temp) = c("orderID", "articleID", "quantity_per_order_article")
df_all = join(df_all, temp, by=c("orderID", "articleID"))
# quantity per orderID and productGroup
temp = aggregate(df_all$quantity, list(df_all$orderID, df_all$productGroup), sum)
names(temp) = c("orderID", "productGroup", "quantity_per_order_prod")
df_all = join(df_all, temp, by=c("orderID", "productGroup"))
# quantity per orderID, productGroup and rrp
temp = aggregate(df_all$quantity, list(df_all$orderID, df_all$productGroup, df_all$rrp), sum)
names(temp) = c("orderID", "productGroup", "rrp", "quantity_per_order_prod_rrp")
df_all = join(df_all, temp, by=c("orderID", "productGroup", "rrp"))

# number of orders per customerID and orderDate
temp = aggregate(df_all$quantity, list(df_all$customerID, df_all$orderDate, df_all$orderID), sum)
names(temp) = c('customerID', 'orderDate', 'orderID', 'quantity_per_customer_orderDate_orderID') 
# it is extremely slow when working on factor or character
temp = aggregate(temp$quantity_per_customer_orderDate_orderID, list(temp$customerID, temp$orderDate), length)
names(temp) = c('customerID', 'orderDate', 'num_of_order_per_customer_orderDate')
df_all = join(df_all, temp, by=c('customerID', 'orderDate'))

# three way table ###############################################################################################################
# quantity per customerID, colorCode and productGroup
temp = aggregate(df_all$quantity, list(df_all$customerID, df_all$colorCode, df_all$productGroup), sum)
names(temp) = c("customerID", "colorCode", "productGroup", "quantity_per_customer_color_prod")
df_all = join(df_all, temp, by=c("customerID", "colorCode", "productGroup"))
# quantity per customerID, sizeCode and productGroup
temp = aggregate(df_all$quantity, list(df_all$customerID, df_all$sizeCode, df_all$productGroup), sum)
names(temp) = c("customerID", "sizeCode", "productGroup", "quantity_per_customer_size_prod")
df_all = join(df_all, temp, by=c("customerID", "sizeCode", "productGroup"))
# quantity per orderID, colorCode and productGroup
temp = aggregate(df_all$quantity, list(df_all$orderID, df_all$colorCode, df_all$productGroup), sum)
names(temp) = c("orderID", "colorCode", "productGroup", "quantity_per_order_color_prod")
df_all = join(df_all, temp, by=c("orderID", "colorCode", "productGroup"))
# quantity per orderID, sizeCode and productGroup
temp = aggregate(df_all$quantity, list(df_all$orderID, df_all$sizeCode, df_all$productGroup), sum)
names(temp) = c("orderID", "sizeCode", "productGroup", "quantity_per_order_size_prod")
df_all = join(df_all, temp, by=c("orderID", "sizeCode", "productGroup"))

# quantity per customerID, colorCode and articleID
temp = aggregate(df_all$quantity, list(df_all$customerID, df_all$colorCode, df_all$articleID), sum)
names(temp) = c("customerID", "colorCode", "articleID", "quantity_per_customer_color_article")
df_all = join(df_all, temp, by=c("customerID", "colorCode", "articleID"))
# quantity per customerID, sizeCode and articleID
temp = aggregate(df_all$quantity, list(df_all$customerID, df_all$sizeCode, df_all$articleID), sum)
names(temp) = c("customerID", "sizeCode", "articleID", "quantity_per_customer_size_article")
df_all = join(df_all, temp, by=c("customerID", "sizeCode", "articleID"))
# quantity per orderID, colorCode and articleID
temp = aggregate(df_all$quantity, list(df_all$orderID, df_all$colorCode, df_all$articleID), sum)
names(temp) = c("orderID", "colorCode", "articleID", "quantity_per_order_color_article")
df_all = join(df_all, temp, by=c("orderID", "colorCode", "articleID"))
# quantity per orderID, sizeCode and articleID
temp = aggregate(df_all$quantity, list(df_all$orderID, df_all$sizeCode, df_all$articleID), sum)
names(temp) = c("orderID", "sizeCode", "articleID", "quantity_per_order_size_article")
df_all = join(df_all, temp, by=c("orderID", "sizeCode", "articleID"))

# quantity per customerID, colorCode, productGroup and rrp
temp = aggregate(df_all$quantity, list(df_all$customerID, df_all$colorCode, df_all$productGroup, df_all$rrp), sum)
names(temp) = c("customerID", "colorCode", "productGroup", "rrp", "quantity_per_customer_color_prod_rrp")
df_all = join(df_all, temp, by=c("customerID", "colorCode", "productGroup", "rrp"))
# quantity per customerID, sizeCode, productGroup and rrp
temp = aggregate(df_all$quantity, list(df_all$customerID, df_all$sizeCode, df_all$productGroup, df_all$rrp), sum)
names(temp) = c("customerID", "sizeCode", "productGroup", "rrp", "quantity_per_customer_size_prod_rrp")
df_all = join(df_all, temp, by=c("customerID", "sizeCode", "productGroup", "rrp"))
# quantity per orderID, colorCode, productGroup and rrp
temp = aggregate(df_all$quantity, list(df_all$orderID, df_all$colorCode, df_all$productGroup, df_all$rrp), sum)
names(temp) = c("orderID", "colorCode", "productGroup", "rrp", "quantity_per_order_color_prod_rrp")
df_all = join(df_all, temp, by=c("orderID", "colorCode", "productGroup", "rrp"))
# quantity per orderID, sizeCode, productGroup and rrp
temp = aggregate(df_all$quantity, list(df_all$orderID, df_all$sizeCode, df_all$productGroup, df_all$rrp), sum)
names(temp) = c("orderID", "sizeCode", "productGroup", "rrp", "quantity_per_order_size_prod_rrp")
df_all = join(df_all, temp, by=c("orderID", "sizeCode", "productGroup", "rrp"))

df_all$quantity_ratio_per_customer_color_prod = df_all$quantity_per_customer_color_prod/df_all$quantity_per_customer_prod
df_all$quantity_ratio_per_customer_size_prod = df_all$quantity_per_customer_size_prod/df_all$quantity_per_customer_prod
df_all$quantity_ratio_per_order_color_prod = df_all$quantity_per_order_color_prod/df_all$quantity_per_order_prod
df_all$quantity_ratio_per_order_size_prod = df_all$quantity_per_order_size_prod/df_all$quantity_per_order_prod

df_all$quantity_ratio_per_customer_color_article = df_all$quantity_per_customer_color_article/df_all$quantity_per_customer_article
df_all$quantity_ratio_per_customer_size_article = df_all$quantity_per_customer_size_article/df_all$quantity_per_customer_article
df_all$quantity_ratio_per_order_color_article = df_all$quantity_per_order_color_article/df_all$quantity_per_order_article
df_all$quantity_ratio_per_order_size_article = df_all$quantity_per_order_size_article/df_all$quantity_per_order_article

df_all$quantity_ratio_per_customer_color_prod_rrp = df_all$quantity_per_customer_color_prod_rrp/df_all$quantity_per_customer_prod_rrp
df_all$quantity_ratio_per_customer_size_prod_rrp = df_all$quantity_per_customer_size_prod_rrp/df_all$quantity_per_customer_prod_rrp
df_all$quantity_ratio_per_order_color_prod_rrp = df_all$quantity_per_order_color_prod_rrp/df_all$quantity_per_order_prod_rrp
df_all$quantity_ratio_per_order_size_prod_rrp = df_all$quantity_per_order_size_prod_rrp/df_all$quantity_per_order_prod_rrp

# sql
temp = sqldf('select customerID, orderDate
             from df_all
             group by customerID, orderDate
             order by customerID, orderDate')
temp = data.table(temp)
temp[, index := 1:.N, by=c("customerID")]
temp[, orderDateNext := as.Date(orderDate[index + 1]), by=c("customerID")]
temp[, orderDatePrev := c(as.Date(NA), as.Date(orderDate[index - 1])), by=c("customerID")]
temp$orderTimeNext = as.integer(as.Date(temp$orderDateNext) - as.Date(temp$orderDate))
temp$orderTimePrev = as.integer(as.Date(temp$orderDatePrev) - as.Date(temp$orderDate))
temp$orderTimeNext[is.na(temp$orderTimeNext)] = 1000
temp$orderTimePrev[is.na(temp$orderTimePrev)] = -1000
df_all = join(df_all, temp[, c("orderTimeNext", "orderDate", "orderTimePrev", "customerID"), with=FALSE],
              by=c("customerID", "orderDate"))

# holidays
temp = unique(df_all$orderDate)
near_holi = sapply(temp, function(x) as.integer(any(abs(as.integer(as.Date(x)-holidays))<7)))
df_all = join(df_all, data.frame(orderDate = temp, near_holi = near_holi), by = c('orderDate'))

# choice item across order
orderDate_dic = tapply(as.Date(df_all$orderDate), df_all$orderID, unique)

choice_item_across_order = function(temp_obj, obj){
  
  df_all$temp = temp_obj
  aggr = aggregate(df_all$quantity, list(df_all$customerID, df_all$temp, df_all$orderID), sum)
  names(aggr) = c("customerID", "temp", "orderID_list", "quantity_list")
  aggr_ind = aggregate(1:nrow(aggr), list(aggr$customerID, aggr$temp), as.numeric)
  names(aggr_ind) = c("customerID", "temp", "orderID_ind")
  orderID_ind = aggr_ind$orderID_ind
  order_count = sapply(orderID_ind, length)
  choice_item_order_number = unlist(sapply(1:length(order_count), function(x) 1:order_count[x]))
  choice_item_order_number_rev = unlist(sapply(1:length(order_count), function(x) order_count[x]:1))
  choice_item_order_index = rep(order_count, times=order_count)
  orderID_ind = unlist(orderID_ind)
  orderID_list = as.character(aggr$orderID_list[orderID_ind])
  names(orderID_list) = aggr_ind$temp[rep(1:length(order_count), times=order_count)]
  
  orderDate_list = orderDate_dic[orderID_list]
  choice_item_order_date_diff_prev = rep(1000, length(orderID_list))
  choice_item_order_date_diff_next = rep(1000, length(orderID_list))
  date_diff = c(0, diff(orderDate_list[choice_item_order_index>1]))[choice_item_order_number[choice_item_order_index>1]>1]
  choice_item_order_date_diff_prev[choice_item_order_number>1] = date_diff
  choice_item_order_date_diff_next[choice_item_order_number_rev>1] = date_diff
  
  aggr = data.frame(temp=names(orderID_list), orderID=orderID_list, choice_item_order_number, choice_item_order_index, choice_item_order_date_diff_prev, choice_item_order_date_diff_next)
  aggr$temp = as.character(aggr$temp)
  names(aggr) = c("temp", "orderID", paste(obj, "_item_order_number", sep=""), paste(obj, "_item_order_index", sep=""), paste(obj, "_item_order_date_diff_prev", sep=""), paste(obj, "_item_order_date_diff_next", sep=""))
  df_all = join(df_all, aggr, by=c("temp", "orderID"))
  df_all = subset(df_all, select=-temp)
  return(df_all)
}

df_all = choice_item_across_order(df_all$articleID, "choice")
df_all = choice_item_across_order(df_all$temp_ac, "ac")
df_all = choice_item_across_order(df_all$temp_as, "as")
df_all = choice_item_across_order(df_all$temp_cp, "cp")
df_all = choice_item_across_order(df_all$temp_sp, "sp")
df_all = choice_item_across_order(df_all$temp_csp, "csp")
df_all = choice_item_across_order(df_all$temp_pr, "pr")
df_all = choice_item_across_order(df_all$temp_cpr, "cpr")
df_all = choice_item_across_order(df_all$temp_spr, "spr")
df_all = choice_item_across_order(df_all$temp_cspr, "cspr")
df_all = choice_item_across_order(df_all$temp_choice_1, "choice_1")
df_all = choice_item_across_order(df_all$temp_as_1, "as_1")
df_all = choice_item_across_order(df_all$temp_sp_1, "sp_1")
df_all = choice_item_across_order(df_all$temp_pr_1, "pr_1")
df_all = choice_item_across_order(df_all$temp_spr_1, "spr_1")
df_all = choice_item_across_order(df_all$temp_choice_234, "choice_234")
df_all = choice_item_across_order(df_all$temp_as_234, "as_234")
df_all = choice_item_across_order(df_all$temp_sp_234, "sp_234")
df_all = choice_item_across_order(df_all$temp_pr_234, "pr_234")
df_all = choice_item_across_order(df_all$temp_spr_234, "spr_234")
df_all = choice_item_across_order(df_all$temp_choice_34, "choice_34")
df_all = choice_item_across_order(df_all$temp_as_34, "as_34")
df_all = choice_item_across_order(df_all$temp_sp_34, "sp_34")
df_all = choice_item_across_order(df_all$temp_pr_34, "pr_34")
df_all = choice_item_across_order(df_all$temp_spr_34, "spr_34")
df_all = choice_item_across_order(df_all$temp_choice_4, "choice_4")
df_all = choice_item_across_order(df_all$temp_as_4, "as_4")
df_all = choice_item_across_order(df_all$temp_sp_4, "sp_4")
df_all = choice_item_across_order(df_all$temp_pr_4, "pr_4")
df_all = choice_item_across_order(df_all$temp_spr_4, "spr_4")
temp = apply(cbind(df_all$articleID, df_all$colorCode, df_all$sizeCode), 1, function(x) paste(x, sep="", collapse=" "))
df_all = choice_item_across_order(temp, "acs")
temp = apply(cbind(df_all$articleID, df_all$colorCode, df_all$sizeCode, df_all$price), 1, function(x) paste(x, sep="", collapse=" "))
df_all = choice_item_across_order(temp, "acsp")

### Minjie's new features
# voucherID numeric
temp = as.numeric(sapply(as.character(df_all$voucherID), function(x) substring(x, 3)))
temp[df_all$voucherID=='-1'] = -1
temp[df_all$voucherID=='0'] = 0
df_all$voucherID = temp

# num of device per customer per date
df_all = data.table(df_all)
df_all[, num_of_device := length(unique(deviceID)), by=c("customerID", "orderDate")]

# num of payment method per customer per date
df_all[, num_of_pay_method := length(unique(paymentMethod)), by=c("customerID", "orderDate")]

# total price per customer per date
df_all[, price_per_cust_date := sum(price), by=c("customerID", "orderDate")]
df_all[, discounted_price_per_cust_date := sum(discounted_price_per_quantity*quantity), by=c("customerID", "orderDate")]

# total price per customer per 3days/5days/7days
df_all$orderDate_ind = as.numeric(df_all$orderDate)
df_all[, price_per_cust_3date := sapply(1:.N, function(x) sum(price[abs(orderDate_ind-orderDate_ind[x])<=1])), by=c("customerID")]
df_all[, price_per_cust_5date := sapply(1:.N, function(x) sum(price[abs(orderDate_ind-orderDate_ind[x])<=2])), by=c("customerID")]
df_all[, price_per_cust_7date := sapply(1:.N, function(x) sum(price[abs(orderDate_ind-orderDate_ind[x])<=3])), by=c("customerID")]
df_all[, price_per_cust_11date := sapply(1:.N, function(x) sum(price[abs(orderDate_ind-orderDate_ind[x])<=5])), by=c("customerID")]
df_all[, price_per_cust_15date := sapply(1:.N, function(x) sum(price[abs(orderDate_ind-orderDate_ind[x])<=7])), by=c("customerID")]

df_all = as.data.frame(df_all)
###

df_all = subset(df_all, select=-c(orderID, voucherID, price, orderDate, orderDate_ind, first_orderDate, rrp_new, choice_item_index, ac_item_index, as_item_index, cp_item_index, sp_item_index, csp_item_index, pr_item_index, cpr_item_index, spr_item_index, cspr_item_index, choice_1_item_index, as_1_item_index, sp_1_item_index, pr_1_item_index, spr_1_item_index, choice_234_item_index, as_234_item_index, sp_234_item_index, pr_234_item_index, spr_234_item_index, choice_34_item_index, as_34_item_index, sp_34_item_index, pr_34_item_index, spr_34_item_index, choice_4_item_index, as_4_item_index, sp_4_item_index, pr_4_item_index, spr_4_item_index, temp_ac, temp_as, temp_cp, temp_sp, temp_csp, temp_pr, temp_cpr, temp_spr, temp_cspr, temp_choice_1, temp_as_1, temp_sp_1, temp_pr_1, temp_spr_1, temp_choice_234, temp_as_234, temp_sp_234, temp_pr_234, temp_spr_234, temp_choice_34, temp_as_34, temp_sp_34, temp_pr_34, temp_spr_34, temp_choice_4, temp_as_4, temp_sp_4, temp_pr_4, temp_spr_4))

df_all = df_all[, !(names(df_all) %in% OHE_feats)]

df_train_quantity = df_train$quantity

save(df_all, y, df_train_quantity, ind_drop, file = "expand_data_old.RData")