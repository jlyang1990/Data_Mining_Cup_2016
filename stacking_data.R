#' combine all the prediction values from 1st layer models and top important features, to make feature dataset in 2nd layer model (model stacking)
#' written by Jilei Yang

#' consider two feature transformation types for new feature items in testing dataset: "new" for entire transformation, "old" for transformation when confident
feature_type <- "new"
# feature_type <- "old"

#' list of models for stacking
model_list <- c("xgb_result_%s_base_90_70",
                "xgb_result_%s_base_90_80",
                "xgb_result_%s_base_90_90",
                "xgb_result_%s_base_95_75",
                "xgb_result_%s_base_95_80",
                "xgb_result_%s_base_95_85",
                "xgb_result_%s_drop_gift_95_80",
                "xgb_result_%s_drop_low_freq_cust_95_80",
                "xgb_result_%s_drop_feat_95_75",
                "xgb_result_%s_drop_feat_95_80",
                "xgb_result_%s_drop_feat_95_85",
                "xgb_result_%s_add_likelihood_cust_95_80",
                "xgb_result_%s_add_likelihood_cust_90_85",
                "xgb_result_%s_add_likelihood_month_95_80",
                "dl_result_%s",
                "rf_result_%s",
                "glm_result_%s"
)

model_list <- sprintf(model_list, feature_type)

#' number of top important features added into feature dataset for stacking
num_imp_feat <- 100

#' generate stacking_data
load(sprintf("feature_data_%s.RData", feature_type))
load(sprintf("%s.RData", model_list[5]))

imp_var <- as.matrix(importance_matrix)[1:num_imp_feat, 1]
imp_feat <- df_all[, (names(df_all) %in% imp_var)]
imp_feat <- as.matrix(imp_feat[rep(1:nrow(df_all), times = df_all$quantity), ])

y_pred_prob_feat_all <- c()
for(model in model_list) {
  load(sprintf("%s.RData", model))
  y_pred_prob_feat_all <- cbind(y_pred_prob_feat_all, y_pred_prob_feat)
}

X_all <- cbind(y_pred_prob_feat_all, imp_feat)

save(X_all, y, y_index, ind_drop, file = sprintf("stacking_data_%s.RData", feature_type))