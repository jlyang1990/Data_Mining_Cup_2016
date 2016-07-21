#' generate all the features (except the likelihood features due to data leakage) for the training and testing dataset
#' written by Jilei Yang, Minjie Fan
#' modified by Jilei Yang

rm(list = ls())
library(caret)
library(stringr)
library(data.table)

#' consider two feature transformation types for new feature items in testing dataset: "new" for entire transformation, "old" for transformation when confident
feature_type <- "new"
# feature_type <- "old"

#' load data
df_train <- fread('orders_train.txt')
df_test <- fread('orders_class.txt')


#' preprocess data ############################################################################################################

#' clean data
#' drop records with quantity=0 and quantity<returnQuantity
df_train <- df_train[quantity > 0 & quantity >= returnQuantity, ]
ind_drop <- which(df_test$quantity == 0)
#' drop records with productGroup=NA and rrp=NA since we expect returnQuantity for these records should be zero
#' as Qi Gao suggested, these records all have articleID i1004001 and they could be add-on items that can not be returned
df_train <- df_train[!is.na(productGroup), ]
ind_drop <- c(ind_drop, which(is.na(df_test$productGroup)))
df_test <- df_test[-ind_drop, ]

#' convert new product group and color (suggested by Qi Gao)
if(feature_type == "new") {
  df_test[productGroup %in% c(1214, 1222)]$productGroup <- 15
  df_test[productGroup %in% c(1220, 1221)]$productGroup <- 14
  df_test[productGroup == 1225]$productGroup <- 5
  df_test[productGroup == 1230]$productGroup <- 4
  df_test[productGroup == 1231]$productGroup <- 3
  df_test[productGroup == 1234]$productGroup <- 8
  df_test[productGroup == 1236]$productGroup <- 1
  df_test[productGroup == 1237]$productGroup <- 2
  df_test[productGroup %in% c(1257, 1258)]$productGroup <- 17
  df_test[productGroup == 1289]$productGroup <- 45
} else {
  df_test[productGroup == 1231]$productGroup <- 3
  df_test[productGroup == 1237]$productGroup <- 2
  df_test[productGroup == 1289]$productGroup <- 45
}

temp <- df_test[colorCode > 9999]$colorCode
df_test[colorCode > 9999]$colorCode <- as.integer(temp %% 1000 + temp %/% 10000 * 1000)

#' drop response and combine training and testing dataset
n_train <- nrow(df_train)
y <- df_train$returnQuantity
df_train <- df_train[, returnQuantity := NULL]
df_train_quantity <- df_train$quantity
df_all <- rbind(df_train, df_test)
rm(df_train, df_test)


#' preliminary feature engineering ############################################################################################

#' convert date to year, month, day, weekday
df_all[, c("order_year", "order_month", "order_day") := data.frame(str_split_fixed(orderDate, '-', 3))]
df_all[, c("order_year", "order_month", "order_day") := .(as.numeric(order_year), as.numeric(order_month), as.numeric(order_day))]
df_all[, order_weekday := weekdays(as.Date(orderDate))]

#' OHE (one hot encoding)
df_all[, c("productGroup", "deviceID") := .(as.character(productGroup), as.character(deviceID))]
OHE_feats <- c('order_weekday', 'sizeCode', 'productGroup', 'deviceID', 'paymentMethod')
OHE_formula <- paste('~', paste(OHE_feats, collapse = ' + '))
dummies <- dummyVars(as.formula(OHE_formula), data = df_all)
df_all_OHE <- data.table(predict(dummies, newdata = df_all))
df_all <- cbind(df_all, df_all_OHE)
rm(df_all_OHE)

#' decipher colorCode (suggested by Nana Wang, Minjie Fan, Jilei Yang)
#' "colorCode_1", "colorCode_2", "colorCode_3" and "colorCode_4" contain the 1st, 2nd, 3rd and 4th digit of colorCode
#' "colorCode_12", "colorCode_34" contain the 1st&2nd and 3rd&4th digits of colorCode
#' "colorCode_123", "colorCode_234" contain the 1st&2nd&3rd and 2nd&3rd&4th digits of colorCode
colorCode_list <- data.table(t(sapply(df_all$colorCode, function(x) as.numeric(substring(x, first = c(1:4, 1, 3, 1, 2), last = c(1:4, 2, 4, 3, 4))))))
names(colorCode_list) <- c("colorCode_1", "colorCode_2", "colorCode_3", "colorCode_4", "colorCode_12", "colorCode_34", "colorCode_123", "colorCode_234")
df_all <- cbind(df_all, colorCode_list)
rm(colorCode_list)

#' decipher sizeCode (suggested by Qi Gao, Hao Ji)
df_all[, sizeNew := sizeCode]
df_all[sizeCode == 'XS']$sizeNew <- '34'
df_all[sizeCode == 'S']$sizeNew <- '36'
df_all[sizeCode == 'M']$sizeNew <- '38'
df_all[sizeCode == 'L']$sizeNew <- '40'
df_all[sizeCode == 'XL']$sizeNew <- '42'
df_all[sizeCode == '75']$sizeNew <- '32'
df_all[sizeCode == '80']$sizeNew <- '34'
df_all[sizeCode == '85']$sizeNew <- '36'
df_all[sizeCode == '90']$sizeNew <- '38'
df_all[sizeCode == '95']$sizeNew <- '40'
df_all[sizeCode == '100']$sizeNew <- '42'

df_all[productGroup == '2' & sizeCode == '34']$sizeNew <- '44'
df_all[productGroup == '2' & sizeCode %in% c('33','32')]$sizeNew <- '42'
df_all[productGroup == '2' & sizeCode == '33']$sizeNew <- '42'
df_all[productGroup == '2' & sizeCode %in% c('31','30')]$sizeNew <- '40'
df_all[productGroup == '2' & sizeCode == '29']$sizeNew <- '38'
df_all[productGroup == '2' & sizeCode %in% c('27','28')]$sizeNew <- '36'
df_all[productGroup == '2' & sizeCode == '26']$sizeNew <- '34'
df_all[productGroup == '2' & sizeCode %in% c('25','24')]$sizeNew <- '32'

df_all[sizeCode %in% c('A', 'I')]$sizeNew <- NA

df_all[, sizeNew := as.numeric(sizeNew)]
df_all[is.na(sizeNew)]$sizeNew <- mean(df_all$sizeNew, na.rm = TRUE)

#' price per quantity: price/quantity
df_all[, price_per_quantity := price / quantity]
#' discounted price per quantity: rrp-price_per_quantity
df_all[, discounted_price_per_quantity := rrp - price_per_quantity]
#' discounted ratio per quantity
df_all[, discounted_ratio_per_quantity := discounted_price_per_quantity / rrp]
df_all[is.na(discounted_ratio_per_quantity)]$discounted_ratio_per_quantity = 0
#' true price per quantity
df_all[, true_price_per_quantity := price_per_quantity * (1 - voucherAmount / tapply(price, orderID, sum)[orderID])]
df_all[is.na(true_price_per_quantity)]$true_price_per_quantity <- 0
#' true discounted price per quantity
df_all[, true_discounted_price_per_quantity := rrp - true_price_per_quantity]
#' true discounted ratio per quantity
df_all[, true_discounted_ratio_per_quantity := true_discounted_price_per_quantity / rrp]
df_all[is.na(true_discounted_ratio_per_quantity)]$true_discounted_ratio_per_quantity <- 0


#' order feature ##############################################################################################################

#' total frequency of order items per order
df_all[, freq_per_order := .N, by = orderID]
#' total quantity per order
df_all[, quantity_per_order := sum(quantity), by = orderID]
#' total number of types of articleID per order
df_all[, article_per_order := length(unique(articleID)), by = orderID]
#' total number of types of productGroup per order
df_all[, prod_per_order := length(unique(productGroup)), by = orderID]
#' total number of types of colorCode per order
df_all[, color_per_order := length(unique(colorCode)), by = orderID]
#' total number of types of sizeCode per order
df_all[, size_per_order := length(unique(sizeCode)), by = orderID]

#' total original price per order
df_all[, original_price_per_order := sum(rrp * quantity), by = orderID]
#' total price per order
df_all[, price_per_order := sum(price), by = orderID]
#' total true price per order
df_all[, true_price_per_order := price_per_order - voucherAmount]
#' total discounted price per order
df_all[, discounted_price_per_order := original_price_per_order - true_price_per_order]
#' total discounted ratio per order
df_all[, discounted_ratio_per_order := discounted_price_per_order / original_price_per_order]
df_all[is.na(discounted_ratio_per_order)]$discounted_ratio_per_order <- 0
#' mean, max, min rrp per order
df_all[, mean_rrp_per_order := mean(rrp), by = orderID]
df_all[, max_rrp_per_order := max(rrp), by = orderID]
df_all[, min_rrp_per_order := min(rrp), by = orderID]

#' function to generate item_quantity_index, item_quantity_per_order and item_quantity_ratio_per_order for choice order item
#' choice order item: items sharing some similarities in articleID, colorCode, sizeCode, productGroup and rrp
ChoiceItemWithinOrder <- function(obj) {
  df_all[, sprintf("%s_item_quantity_index", obj) := .N, by = .(eval(parse(text = sprintf("temp_%s", obj))), orderID)]
  df_all[, sprintf("%s_item_quantity_per_order", obj) := sum(quantity * (eval(parse(text = sprintf("%s_item_quantity_index", obj))) > 1)), by = orderID]
  df_all[, sprintf("%s_item_quantity_ratio_per_order", obj) := eval(parse(text = sprintf("%s_item_quantity_per_order", obj))) / quantity_per_order]
  return(df_all)
}

#' article item: items with the same articleID
df_all[, temp_article := articleID]
#' ac item: items with the same articleID and colorCode
df_all[, temp_ac := apply(cbind(articleID, colorCode), 1, function(x) paste(x, sep = "", collapse = " "))]
#' as item: items with the same articleID and sizeCode
df_all[, temp_as := apply(cbind(articleID, sizeCode), 1, function(x) paste(x, sep = "", collapse = " "))]
#' cp item: items with the same productGroup and colorCode
df_all[, temp_cp := apply(cbind(colorCode, productGroup), 1, function(x) paste(x, sep = "", collapse = " "))]
#' sp item: items with the same productGroup and sizeCode
df_all[, temp_sp := apply(cbind(sizeCode, productGroup), 1, function(x) paste(x, sep = "", collapse = " "))]
#' csp item: items with the same productGroup, colorCode and sizeCode
df_all[, temp_csp := apply(cbind(colorCode, sizeCode, productGroup), 1, function(x) paste(x, sep = "", collapse = " "))]
#' make cluster of rrp for choice order items
df_all[, rrp_new := rrp]
df_all[rrp_new == 9.95]$rrp_new = 9.99
df_all[rrp_new == 12.95]$rrp_new = 12.99
df_all[rrp_new %in% c(15, 15.95)]$rrp_new = 15.99
df_all[rrp_new == 19.95]$rrp_new = 19.99
df_all[rrp_new == 29.95]$rrp_new = 29.99
df_all[rrp_new == 35]$rrp_new = 35.99
df_all[rrp_new %in% c(39.95, 40)]$rrp_new = 39.99
df_all[rrp_new == 49.95]$rrp_new = 49.99
df_all[rrp_new == 59.95]$rrp_new = 59.99
df_all[rrp_new == 69.95]$rrp_new = 69.99
#' pr item: items with the same rrp and productGroup
df_all[, temp_pr := apply(cbind(productGroup, rrp_new), 1, function(x) paste(x, sep = "", collapse = " "))]
#' cpr item: items with the same rrp, productGroup and colorCode
df_all[, temp_cpr := apply(cbind(colorCode, productGroup, rrp_new), 1, function(x) paste(x, sep = "", collapse = " "))]
#' spr item: items with the same rrp, productGroup and sizeCode
df_all[, temp_spr := apply(cbind(sizeCode, productGroup, rrp_new), 1, function(x) paste(x, sep = "", collapse = " "))]
#' cspr item: items with the same rrp, productGroup, colorCode and sizeCode
df_all[, temp_cspr := apply(cbind(colorCode, sizeCode, productGroup, rrp_new), 1, function(x) paste(x, sep = "", collapse = " "))]
#' article_1 item: items with the same articleID and colorCode_1
df_all[, temp_article_1 := apply(cbind(articleID, colorCode_1), 1, function(x) paste(x, sep = "", collapse = " "))]
#' as_1 item: items with the same articleID, sizeCode and colorCode_1
df_all[, temp_as_1 := apply(cbind(articleID, sizeCode, colorCode_1), 1, function(x) paste(x, sep = "", collapse = " "))]
#' sp_1 item: items with the same productGroup, sizeCode and colorCode_1
df_all[, temp_sp_1 := apply(cbind(sizeCode, productGroup, colorCode_1), 1, function(x) paste(x, sep = "", collapse = " "))]
#' pr_1 item: items with the same rrp and productGroup and colorCode_1
df_all[, temp_pr_1 := apply(cbind(productGroup, rrp_new, colorCode_1), 1, function(x) paste(x, sep = "", collapse = " "))]
#' spr_1 item: items with the same rrp, productGroup, sizeCode and colorCode_1
df_all[, temp_spr_1 := apply(cbind(sizeCode, productGroup, rrp_new, colorCode_1), 1, function(x) paste(x, sep = "", collapse = " "))]
#' article_234 item: items with the same articleID and colorCode_234
df_all[, temp_article_234 := apply(cbind(articleID, colorCode_234), 1, function(x) paste(x, sep = "", collapse = " "))]
#' as_234 item: items with the same articleID, sizeCode and colorCode_234
df_all[, temp_as_234 := apply(cbind(articleID, sizeCode, colorCode_234), 1, function(x) paste(x, sep = "", collapse = " "))]
#' sp_234 item: items with the same productGroup, sizeCode and colorCode_234
df_all[, temp_sp_234 := apply(cbind(sizeCode, productGroup, colorCode_234), 1, function(x) paste(x, sep = "", collapse = " "))]
#' pr_234 item: items with the same rrp, productGroup and colorCode_234
df_all[, temp_pr_234 := apply(cbind(productGroup, rrp_new, colorCode_234), 1, function(x) paste(x, sep = "", collapse = " "))]
#' spr_234 item: items with the same rrp, productGroup, sizeCode and colorCode_234
df_all[, temp_spr_234 := apply(cbind(sizeCode, productGroup, rrp_new, colorCode_234), 1, function(x) paste(x, sep = "", collapse = " "))]
#' article_34 item: items with the same articleID and colorCode_34
df_all[, temp_article_34 := apply(cbind(articleID, colorCode_34), 1, function(x) paste(x, sep = "", collapse = " "))]
#' as_34 item: items with the same articleID, sizeCode and colorCode_34
df_all[, temp_as_34 := apply(cbind(articleID, sizeCode, colorCode_34), 1, function(x) paste(x, sep = "", collapse = " "))]
#' sp_34 item: items with the same productGroup, sizeCode and colorCode_34
df_all[, temp_sp_34 := apply(cbind(sizeCode, productGroup, colorCode_34), 1, function(x) paste(x, sep = "", collapse = " "))]
#' pr_34 item: items with the same rrp, productGroup and colorCode_34
df_all[, temp_pr_34 := apply(cbind(productGroup, rrp_new, colorCode_34), 1, function(x) paste(x, sep = "", collapse = " "))]
#' spr_34 item: items with the same rrp, productGroup, sizeCode and colorCode_34
df_all[, temp_spr_34 := apply(cbind(sizeCode, productGroup, rrp_new, colorCode_34), 1, function(x) paste(x, sep = "", collapse = " "))]
#' article_4 item: items with the same articleID and colorCode_4
df_all[, temp_article_4 := apply(cbind(articleID, colorCode_4), 1, function(x) paste(x, sep = "", collapse = " "))]
#' as_4 item: items with the same articleID, sizeCode and colorCode_4
df_all[, temp_as_4 := apply(cbind(articleID, sizeCode, colorCode_4), 1, function(x) paste(x, sep = "", collapse = " "))]
#' sp_4 item: items with the same productGroup, sizeCode and colorCode_4
df_all[, temp_sp_4 := apply(cbind(sizeCode, productGroup, colorCode_4), 1, function(x) paste(x, sep = "", collapse = " "))]
#' pr_4 item: items with the same rrp, productGroup and colorCode_4
df_all[, temp_pr_4 := apply(cbind(productGroup, rrp_new, colorCode_4), 1, function(x) paste(x, sep = "", collapse = " "))]
#' spr_4 item: items with the same rrp, productGroup, sizeCode and colorCode_4
df_all[, temp_spr_4 := apply(cbind(sizeCode, productGroup, rrp_new, colorCode_4), 1, function(x) paste(x, sep = "", collapse = " "))]
#' acs item: item with the same articleID, colorCode and sizeCode
df_all[, temp_acs := apply(cbind(articleID, colorCode, sizeCode), 1, function(x) paste(x, sep="", collapse=" "))]
#' acsp item: item with the same articleID, colorCode, sizeCode and price
df_all[, temp_acsp := apply(cbind(articleID, colorCode, sizeCode, price), 1, function(x) paste(x, sep="", collapse=" "))]

#' choice order item list
choice_item_list <- c("article", "ac", "as", "cp", "sp", "csp", "pr", "cpr", "spr", "cspr", 
                      "article_1", "as_1", "sp_1", "pr_1", "spr_1", 
                      "article_234", "as_234", "sp_234", "pr_234", "spr_234", 
                      "article_34", "as_34", "sp_34", "pr_34", "spr_34", 
                      "article_4", "as_4", "sp_4", "pr_4", "spr_4")

#' implement ChoiceItemWithinOrder to choice order item list
for(choice_item in choice_item_list) {
  df_all <- ChoiceItemWithinOrder(choice_item)
  cat(sprintf("Complete ChoiceItemWithinOrder(\"%s\")", choice_item), "\n")
}


#' customer feature ###########################################################################################################

#' total number of orders per customer
df_all[, order_per_customer := length(unique(orderID)), by = customerID]
#' total frequency of order items per customer
df_all[, freq_per_customer := .N, by = customerID]
#' total quantity per customer
df_all[, quantity_per_customer := sum(quantity), by = customerID]
#' total number of types of articleID per customer
df_all[, article_per_customer := length(unique(articleID)), by = customerID]
#' total number of types of productGroup per customer
df_all[, prod_per_customer := length(unique(productGroup)), by = customerID]
#' total number of types of colorCode per customer
df_all[, colorCode_per_customer := length(unique(colorCode)), by = customerID]
#' total number of types of sizeCode per customer
df_all[, sizeCode_per_customer := length(unique(sizeCode)), by = customerID]
#' total number of types of deviceID per customer
df_all[, device_per_customer := length(unique(deviceID)), by = customerID]
#' total number of types of payment per customer
df_all[, payment_per_customer := length(unique(paymentMethod)), by = customerID]

#' total original price per customer
df_all[, original_price_per_customer := sum(rrp * quantity), by = customerID]
#' total price per customer
df_all[, price_per_customer := sum(price), by = customerID]
#' total quantity of vouchers per customer
df_all[is.na(voucherID)]$voucherID <- "0"
df_all[, voucher_quantity_per_customer := length(unique(orderID[voucherID != "0"])), by = customerID]
#' total voucherAmount per customer
df_all[, voucherAmount_per_customer := sum(voucherAmount[!duplicated(orderID)]), by = customerID]
#' total true price per customer
df_all[, true_price_per_customer := price_per_customer - voucherAmount_per_customer]
#' total discounted price per customer
df_all[, discounted_price_per_customer := original_price_per_customer - true_price_per_customer]
#' discounted ratio per customer
df_all[, discounted_ratio_per_customer := discounted_price_per_customer / original_price_per_customer]
df_all[is.na(discounted_ratio_per_customer)]$discounted_ratio_per_customer <- 0
#' mean, max, min rrp per customer
df_all[, mean_rrp_per_customer := mean(rrp), by = customerID]
df_all[, max_rrp_per_customer := max(rrp), by = customerID]
df_all[, min_rrp_per_customer := min(rrp), by = customerID]

#' function to generate total quantity of choice order items per customer, and ratio of quantity of choice order items per customer
ChoiceItemPerCustomer <- function(obj) {
  #' total quantity of choice order items per customer
  df_all[, sprintf("%s_item_quantity_per_customer", obj) := sum(quantity * (eval(parse(text = sprintf("%s_item_quantity_index", obj))) > 1)), by = customerID]
  #' ratio of quantity of choice order items per customer
  df_all[, sprintf("%s_item_quantity_ratio_per_customer", obj) := eval(parse(text = sprintf("%s_item_quantity_per_customer", obj))) / quantity_per_customer]
  return(df_all)
}

#' implement ChoiceItemPerCustomer to choice order item list
for(choice_item in choice_item_list) {
  df_all <- ChoiceItemPerCustomer(choice_item)
  cat(sprintf("Complete ChoiceItemPerCustomer(\"%s\")", choice_item), "\n")
}

#' mean, max and min frequency of a single article per customer
MySummary <- function(vec) {
  table_vec <- table(vec)
  list(mean = mean(table_vec), max = max(table_vec), min = min(table_vec))
}
df_all[, c("mean_article_freq_per_customer", "max_article_freq_per_customer", "min_article_freq_per_customer") := MySummary(articleID), by = customerID]
#' mean, max and min frequency of a single productGroup per customer
df_all[, c("mean_prod_freq_per_customer", "max_prod_freq_per_customer", "min_prod_freq_per_customer") := MySummary(productGroup), by = customerID]
#' mean, max and min frequency of a single colorCode per customer
df_all[, c("mean_color_freq_per_customer", "max_color_freq_per_customer", "min_color_freq_per_customer") := MySummary(colorCode), by = customerID]
#' mean, max and min frequency of a single sizeCode per customer
df_all[, c("mean_size_freq_per_customer", "max_size_freq_per_customer", "min_size_freq_per_customer") := MySummary(sizeCode), by = customerID]


#' customer-order feature #####################################################################################################

#' mean, max and min frequency in a single order per customer
df_all[, mean_order_freq_per_customer := freq_per_customer / order_per_customer]
df_all[, max_order_freq_per_customer := max(freq_per_order), by = customerID]
df_all[, min_order_freq_per_customer := min(freq_per_order), by = customerID]
#' mean, max and min quantity in a single order per customer
df_all[, mean_order_quantity_per_customer := quantity_per_customer / order_per_customer]
df_all[, max_order_quantity_per_customer := max(quantity_per_order), by = customerID]
df_all[, min_order_quantity_per_customer := min(quantity_per_order), by = customerID]
#' mean, max and min number of types of articleID in a single order per customer
df_all[, mean_order_article_per_customer := mean(article_per_order[!duplicated(orderID)]), by = customerID]
df_all[, max_order_article_per_customer := max(article_per_order), by = customerID]
df_all[, min_order_article_per_customer := min(article_per_order), by = customerID]
#' mean, max and min number of types of productGroup in a single order per customer
df_all[, mean_order_prod_per_customer := mean(prod_per_order[!duplicated(orderID)]), by = customerID]
df_all[, max_order_prod_per_customer := max(prod_per_order), by = customerID]
df_all[, min_order_prod_per_customer := min(prod_per_order), by = customerID]
#' mean, max and min number of types of colorCode in a single order per customer
df_all[, mean_order_color_per_customer := mean(color_per_order), by = customerID]
df_all[, max_order_color_per_customer := max(color_per_order), by = customerID]
df_all[, min_order_color_per_customer := min(color_per_order), by = customerID]
#' mean, max and min number of types of sizeCode in a single order per customer
df_all[, mean_order_size_per_customer := mean(size_per_order), by = customerID]
df_all[, max_order_size_per_customer := max(size_per_order), by = customerID]
df_all[, min_order_size_per_customer := min(size_per_order), by = customerID]

#' mean, max and min original price of a single order per customer
df_all[, mean_order_original_price_per_customer := original_price_per_customer / order_per_customer]
df_all[, max_order_original_price_per_customer := max(original_price_per_order), by = customerID]
df_all[, min_order_original_price_per_customer := min(original_price_per_order), by = customerID]
#' mean, max and min price of a single order per customer
df_all[, mean_order_price_per_customer := price_per_customer / order_per_customer]
df_all[, max_order_price_per_customer := max(price_per_order), by = customerID]
df_all[, min_order_price_per_customer := min(price_per_order), by = customerID]
#' mean, max and min voucherAmount of a single order per customer
df_all[, mean_order_voucherAmount_per_customer := voucherAmount_per_customer / order_per_customer]
df_all[, max_order_voucherAmount_per_customer := max(voucherAmount), by = customerID]
df_all[, min_order_voucherAmount_per_customer := min(voucherAmount), by = customerID]
#' mean, max and min true price of a single order per customer
df_all[, mean_order_true_price_per_customer := true_price_per_customer / order_per_customer]
df_all[, max_order_true_price_per_customer := max(true_price_per_order), by = customerID]
df_all[, min_order_true_price_per_customer := min(true_price_per_order), by = customerID]
#' mean, max and min discounted price of a single order per customer
df_all[, mean_order_discounted_price_per_customer := discounted_price_per_customer / order_per_customer]
df_all[, max_order_discounted_price_per_customer := max(discounted_price_per_order), by = customerID]
df_all[, min_order_discounted_price_per_customer := min(discounted_price_per_order), by = customerID]
#' max and min discounted ratio of a single order per customer
df_all[, max_order_discounted_ratio_per_customer := max(discounted_ratio_per_order), by = customerID]
df_all[, min_order_discounted_ratio_per_customer := min(discounted_ratio_per_order), by = customerID]

#' function to generate mean, max and min quantity of choice order items in a single order per customer, and max and min quantity ratio of choice order items in a single order per customer
ChoiceItemWithinOrderPerCustomer <- function(obj) {
  #' mean, max and min quantity of choice order items in a single order per customer
  df_all[, sprintf("mean_order_%s_item_quantity_per_customer", obj) := eval(parse(text = sprintf("%s_item_quantity_per_customer", obj))) / order_per_customer]
  df_all[, sprintf("max_order_%s_item_quantity_per_customer", obj) := max(eval(parse(text = sprintf("%s_item_quantity_per_order", obj)))), by = customerID]
  df_all[, sprintf("min_order_%s_item_quantity_per_customer", obj) := min(eval(parse(text = sprintf("%s_item_quantity_per_order", obj)))), by = customerID]
  #' max and min quantity ratio of choice order items in a single order per customer
  df_all[, sprintf("max_order_%s_item_quantity_ratio_per_customer", obj) := max(eval(parse(text = sprintf("%s_item_quantity_ratio_per_order", obj)))), by = customerID]
  df_all[, sprintf("min_order_%s_item_quantity_ratio_per_customer", obj) := min(eval(parse(text = sprintf("%s_item_quantity_ratio_per_order", obj)))), by = customerID]
  return(df_all)
}

#' implement ChoiceItemWithinOrderPerCustomer to choice order item list
for(choice_item in choice_item_list) {
  df_all <- ChoiceItemWithinOrderPerCustomer(choice_item)
  cat(sprintf("Complete ChoiceItemWithinOrderPerCustomer(\"%s\")", choice_item), "\n")
}


#' article feature ############################################################################################################

#' total frequency of order items per article
df_all[, freq_per_article := .N, by = articleID]
#' total quantity per article
df_all[, quantity_per_article := sum(quantity), by = articleID]
#' total number of orders per article
df_all[, order_per_article := length(unique(orderID)), by = articleID]
#' total number of customers per article
df_all[, customer_per_article := length(unique(customerID)), by = articleID]

#' mean, max and min of rrp per article
df_all[, mean_rrp_per_article := mean(rrp), by = articleID]
df_all[, max_rrp_per_article := max(rrp), by = articleID]
df_all[, min_rrp_per_article := min(rrp), by = articleID]
#' mean, max and min of discounted price per article
df_all[, mean_discounted_price_per_article := mean(discounted_price_per_quantity), by = articleID]
df_all[, max_discounted_price_per_article := max(discounted_price_per_quantity), by = articleID]
df_all[, min_discounted_price_per_article := min(discounted_price_per_quantity), by = articleID]
#' mean, max and min of discounted ratio per article
df_all[, mean_discounted_ratio_per_article := mean(discounted_ratio_per_quantity), by = articleID]
df_all[, max_discounted_ratio_per_article := max(discounted_ratio_per_quantity), by = articleID]
df_all[, min_discounted_ratio_per_article := min(discounted_ratio_per_quantity), by = articleID]

#' orderDate feature ##########################################################################################################

#' indicator of closeness to holidays
holidays <- as.Date(c('2014-01-01', '2014-04-21', '2014-05-01', '2014-05-29', '2014-06-09', '2014-10-03', '2014-12-25', '2014-12-26', 
                      '2015-01-01', '2015-04-06', '2015-05-01', '2015-05-14', '2015-05-25', '2015-10-03', '2015-12-25', '2015-12-26'))
df_all[, near_holi := as.integer(any(abs(as.Date(orderDate) - holidays) < 7)), by = orderDate]

#' first order date per customer
df_all[, first_orderDate := min(orderDate), by = customerID]
df_all[, c("first_order_year", "first_order_month", "first_order_day") := data.frame(str_split_fixed(first_orderDate, '-', 3))]
df_all[, c("first_order_year", "first_order_month", "first_order_day") := .(as.numeric(first_order_year), as.numeric(first_order_month), as.numeric(first_order_day))]

#' time difference between orders per customer
df_all[, orderDate_int := as.integer(as.Date(orderDate))]
temp <- df_all[, .(orderDate_int = unique(orderDate_int)), by = .(customerID, orderID)]
temp[, orderDate_diff_next := c(diff(orderDate_int), 1000), by = customerID]
temp[, orderDate_diff_prev := c(1000, diff(orderDate_int)), by = customerID]
temp[, c("orderDate_int", "customerID") := NULL]
df_all <- merge(df_all, temp, all.x = TRUE, by = "orderID", sort = FALSE)
rm(temp)

#' function to generate time difference between choice order items across order per customer
ChoiceItemAcrossOrder <- function(obj) {
  temp <- df_all[, .(orderDate_int = unique(orderDate_int)), by = .(customerID, eval(parse(text = sprintf("temp_%s", obj))), orderID)]
  colnames(temp)[colnames(temp) == "parse"] <- sprintf("temp_%s", obj)
  temp[, sprintf("%s_item_order_quantity_count", obj) := 1:.N, by = .(customerID, eval(parse(text = sprintf("temp_%s", obj))))]
  temp[, sprintf("%s_item_order_quantity_index", obj) := .N, by = .(customerID, eval(parse(text = sprintf("temp_%s", obj))))]
  temp[, sprintf("%s_item_orderDate_diff_next", obj) := c(diff(orderDate_int), 1000), by = .(customerID, eval(parse(text = sprintf("temp_%s", obj))))]
  temp[, sprintf("%s_item_orderDate_diff_prev", obj) := c(1000, diff(orderDate_int)), by = .(customerID, eval(parse(text = sprintf("temp_%s", obj))))]
  temp[, c("orderDate_int", "customerID") := NULL]
  merge(df_all, temp, all.x = TRUE, by = c("orderID", (sprintf("temp_%s", obj))), sort = FALSE)
}

#' implement ChoiceItemAcrossOrder to choice order item list
for(choice_item in c(choice_item_list, "acs", "acsp")) {
  df_all <- ChoiceItemAcrossOrder(choice_item)
  cat(sprintf("Complete ChoiceItemAcrossOrder(\"%s\")", choice_item), "\n")
}

#' num of device per customer per date
df_all[, device_per_customer_date := length(unique(deviceID)), by = .(customerID, orderDate)]
#' num of payment method per customer per date
df_all[, payment_per_customer_date := length(unique(paymentMethod)), by = .(customerID, orderDate)]

#' total price per customer per date
df_all[, price_per_customer_date := sum(price), by = .(customerID, orderDate)]
df_all[, discounted_price_per_customer_date := sum(discounted_price_per_quantity * quantity), by = .(customerID, orderDate)]

#' total price per customer per 3days/5days/7days/11days/15days
df_all[, price_per_customer_3date := sapply(1:.N, function(x) sum(price[abs(orderDate_int - orderDate_int[x]) <= 1])), by = customerID]
df_all[, price_per_customer_5date := sapply(1:.N, function(x) sum(price[abs(orderDate_int - orderDate_int[x]) <= 2])), by = customerID]
df_all[, price_per_customer_7date := sapply(1:.N, function(x) sum(price[abs(orderDate_int - orderDate_int[x]) <= 3])), by = customerID]
df_all[, price_per_customer_11date := sapply(1:.N, function(x) sum(price[abs(orderDate_int - orderDate_int[x]) <= 5])), by = customerID]
df_all[, price_per_customer_15date := sapply(1:.N, function(x) sum(price[abs(orderDate_int - orderDate_int[x]) <= 7])), by = customerID]

#' payment feature ############################################################################################################

#' classify payment method into three groups: invoice, rightaway and others
payment_new <- df_all$paymentMethod
payment_new[payment_new %in% c("BPRG", "KGRG", "RG", "BPPL")] <- "INVOICE"
payment_new[payment_new %in% c("PAYPALVC", "CBA", "BPLS", "KKE")] <- "RIGHTAWAY"
df_all[, invoice_ind := payment_new=="INVOICE"]
df_all[, rightaway_ind := payment_new=="RIGHTAWAY"]

#' two way table ##############################################################################################################

#' quantity per customerID and articleID
df_all[, quantity_per_customer_article := sum(quantity), by = .(customerID, articleID)]
#' quantity per customerID and productGroup
df_all[, quantity_per_customer_prod := sum(quantity), by = .(customerID, productGroup)]
#' quantity per customerID, productGroup and rrp
df_all[, quantity_per_customer_prod_rrp := sum(quantity), by = .(customerID, productGroup, rrp)]
#' quantity per customerID and colorCode
df_all[, quantity_per_customer_color := sum(quantity), by = .(customerID, colorCode)]
#' quantity per customerID and sizeCode
df_all[, quantity_per_customer_size := sum(quantity), by = .(customerID, sizeCode)]

#' price per customerID and articleID
df_all[, price_per_customer_article := sum(price), by = .(customerID, articleID)]
#' price per customerID and productGroup
df_all[, price_per_customer_prod := sum(price), by = .(customerID, productGroup)]
#' price per customerID, productGroup and rrp
df_all[, price_per_customer_prod_rrp := sum(price), by = .(customerID, productGroup, rrp)]
#' price per customerID and colorCode
df_all[, price_per_customer_color := sum(price), by = .(customerID, colorCode)]
#' price per customerID and sizeCode
df_all[, price_per_customer_size := sum(price), by = .(customerID, sizeCode)]

#' discounted price per customerID and articleID
df_all[, discounted_price_per_customer_article := sum(discounted_price_per_quantity * quantity), by = .(customerID, articleID)]
#' discounted price per customerID and productGroup
df_all[, discounted_price_per_customer_prod := sum(discounted_price_per_quantity * quantity), by = .(customerID, productGroup)]
#' discounted price per customerID, productGroup and rrp
df_all[, discounted_price_per_customer_prod_rrp := sum(discounted_price_per_quantity * quantity), by = .(customerID, productGroup, rrp)]
#' discounted price per customerID and colorCode
df_all[, discounted_price_per_customer_color := sum(discounted_price_per_quantity * quantity), by = .(customerID, colorCode)]
#' discounted price per customerID and sizeCode
df_all[, discounted_price_per_customer_size := sum(discounted_price_per_quantity * quantity), by = .(customerID, sizeCode)]

#' quantity per orderID and articleID
df_all[, quantity_per_order_article := sum(quantity), by = .(orderID, articleID)]
#' quantity per orderID and productGroup
df_all[, quantity_per_order_prod := sum(quantity), by = .(orderID, productGroup)]
#' quantity per orderID, productGroup and rrp
df_all[, quantity_per_order_prod_rrp := sum(quantity), by = .(orderID, productGroup, rrp)]

#' number of orders per customerID and orderDate
df_all[, order_per_customer_orderDate := length(unique(orderID)), by = .(customerID, orderDate)]

#' three way table ############################################################################################################

#' quantity per customerID, colorCode and productGroup
df_all[, quantity_per_customer_color_prod := sum(quantity), by = .(customerID, colorCode, productGroup)]
#' quantity per customerID, sizeCode and productGroup
df_all[, quantity_per_customer_size_prod := sum(quantity), by = .(customerID, sizeCode, productGroup)]
#' quantity per orderID, colorCode and productGroup
df_all[, quantity_per_order_color_prod := sum(quantity), by = .(orderID, colorCode, productGroup)]
#' quantity per orderID, sizeCode and productGroup
df_all[, quantity_per_order_size_prod:= sum(quantity), by = .(orderID, sizeCode, productGroup)]

#' quantity per customerID, colorCode and articleID
df_all[, quantity_per_customer_color_article := sum(quantity), by = .(customerID, colorCode, articleID)]
#' quantity per customerID, sizeCode and articleID
df_all[, quantity_per_customer_size_article := sum(quantity), by = .(customerID, sizeCode, articleID)]
#' quantity per orderID, colorCode and articleID
df_all[, quantity_per_order_color_article := sum(quantity), by = .(orderID, colorCode, articleID)]
#' quantity per orderID, sizeCode and articleID
df_all[, quantity_per_order_size_article := sum(quantity), by = .(orderID, sizeCode, articleID)]

#' quantity per customerID, colorCode, productGroup and rrp
df_all[, quantity_per_customer_color_prod_rrp := sum(quantity), by = .(customerID, colorCode, productGroup, rrp)]
#' quantity per customerID, sizeCode, productGroup and rrp
df_all[, quantity_per_customer_size_prod_rrp := sum(quantity), by = .(customerID, sizeCode, productGroup, rrp)]
#' quantity per orderID, colorCode, productGroup and rrp
df_all[, quantity_per_order_color_prod_rrp := sum(quantity), by = .(orderID, colorCode, productGroup, rrp)]
#' quantity per orderID, sizeCode, productGroup and rrp
df_all[, quantity_per_order_size_prod_rrp := sum(quantity), by = .(orderID, sizeCode, productGroup, rrp)]

#' quantity ratio
df_all[, quantity_ratio_per_customer_color_prod := quantity_per_customer_color_prod / quantity_per_customer_prod]
df_all[, quantity_ratio_per_customer_size_prod := quantity_per_customer_size_prod / quantity_per_customer_prod]
df_all[, quantity_ratio_per_order_color_prod := quantity_per_order_color_prod / quantity_per_order_prod]
df_all[, quantity_ratio_per_order_size_prod := quantity_per_order_size_prod / quantity_per_order_prod]

df_all[, quantity_ratio_per_customer_color_article := quantity_per_customer_color_article / quantity_per_customer_article]
df_all[, quantity_ratio_per_customer_size_article := quantity_per_customer_size_article / quantity_per_customer_article]
df_all[, quantity_ratio_per_order_color_article := quantity_per_order_color_article / quantity_per_order_article]
df_all[, quantity_ratio_per_order_size_article := quantity_per_order_size_article / quantity_per_order_article]

df_all[, quantity_ratio_per_customer_color_prod_rrp := quantity_per_customer_color_prod_rrp / quantity_per_customer_prod_rrp]
df_all[, quantity_ratio_per_customer_size_prod_rrp := quantity_per_customer_size_prod_rrp / quantity_per_customer_prod_rrp]
df_all[, quantity_ratio_per_order_color_prod_rrp := quantity_per_order_color_prod_rrp / quantity_per_order_prod_rrp]
df_all[, quantity_ratio_per_order_size_prod_rrp := quantity_per_order_size_prod_rrp / quantity_per_order_prod_rrp]


#' save transformed data ######################################################################################################

#' customerID contains the information of the date when the account was created
#' convert articleID, customerID to numeric
df_all[, articleID := as.integer(substring(articleID, 3))]
df_all[, customerID := as.integer(substring(customerID, 3))]

#' delete unused features
df_all[, c("orderID", "price", "voucherID", "orderDate", "orderDate_int", "first_orderDate", "rrp_new",
           "temp_article", "temp_ac", "temp_as", "temp_cp", "temp_sp", "temp_csp", "temp_pr", "temp_cpr", "temp_spr", "temp_cspr",
           "temp_article_1", "temp_as_1", "temp_sp_1", "temp_pr_1", "temp_spr_1",
           "temp_article_234", "temp_as_234", "temp_sp_234", "temp_pr_234", "temp_spr_234",
           "temp_article_34", "temp_as_34", "temp_sp_34", "temp_pr_34", "temp_spr_34",
           "temp_article_4", "temp_as_4", "temp_sp_4", "temp_pr_4", "temp_spr_4",
           "temp_acs", "temp_acsp", OHE_feats) := NULL]

save(df_all, y, df_train_quantity, ind_drop, file = sprintf("feature_data_%s.RData", feature_type))