#' combine all the prediction values from 1st layer models and top important features, to make feature dataset in 2nd layer model (model stacking)
#' written by Jilei Yang

#generate stacking_new##########################

imp_var = as.matrix(importance_matrix)[1:100, 1]
imp_feat = df_all[, (names(df_all) %in% imp_var)]
imp_feat = imp_feat[rep(1:nrow(df_all), times=df_all$quantity), ]
imp_feat = as.matrix(imp_feat)
imp_feat_new = imp_feat

y_pred_prob_feat_xgb_baseline_new_ss90_cs70 = y_pred_prob_feat
y_pred_prob_feat_xgb_baseline_new_ss90_cs80 = y_pred_prob_feat
y_pred_prob_feat_xgb_baseline_new_ss90_cs90 = y_pred_prob_feat
y_pred_prob_feat_xgb_baseline_new_ss95_cs75 = y_pred_prob_feat
y_pred_prob_feat_xgb_baseline_new_ss95_cs80 = y_pred_prob_feat
y_pred_prob_feat_xgb_baseline_new_ss95_cs85 = y_pred_prob_feat
y_pred_prob_feat_xgb_drop_gift_new_ss95_cs80 = y_pred_prob_feat
y_pred_prob_feat_xgb_drop_lowfreq_cust_new_ss95_cs80 = y_pred_prob_feat
y_pred_prob_feat_xgb_drop_feat_new_ss95_cs75 = y_pred_prob_feat
y_pred_prob_feat_xgb_drop_feat_new_ss95_cs80 = y_pred_prob_feat
y_pred_prob_feat_xgb_drop_feat_new_ss95_cs85 = y_pred_prob_feat
y_pred_prob_feat_xgb_likelihood_cust_new_ss95_cs80 = y_pred_prob_feat
y_pred_prob_feat_xgb_likelihood_cust_new_ss90_cs85 = y_pred_prob_feat
y_pred_prob_feat_xgb_likelihood_month_new_ss95_cs80 = y_pred_prob_feat
y_pred_prob_feat_xgb_new = c(y_pred_prob_feat_xgb_baseline_new_ss90_cs70, y_pred_prob_feat_xgb_baseline_new_ss90_cs80, y_pred_prob_feat_xgb_baseline_new_ss90_cs90, y_pred_prob_feat_xgb_baseline_new_ss95_cs75, y_pred_prob_feat_xgb_baseline_new_ss95_cs80, y_pred_prob_feat_xgb_baseline_new_ss95_cs85, y_pred_prob_feat_xgb_drop_gift_new_ss95_cs80, y_pred_prob_feat_xgb_drop_lowfreq_cust_new_ss95_cs80, y_pred_prob_feat_xgb_drop_feat_new_ss95_cs75, y_pred_prob_feat_xgb_drop_feat_new_ss95_cs80, y_pred_prob_feat_xgb_drop_feat_new_ss95_cs85, y_pred_prob_feat_xgb_likelihood_cust_new_ss95_cs80, y_pred_prob_feat_xgb_likelihood_cust_new_ss90_cs85, y_pred_prob_feat_xgb_likelihood_month_new_ss95_cs80)

y_pred_prob_feat_dl_new = y_pred_prob_feat_testnew_DL
y_pred_prob_feat_glm_new = y_pred_prob_feat_test_GLMnet

X_all = cbind(y_pred_prob_feat_xgb_new, y_pred_prob_feat_glm_new, y_pred_prob_feat_dl_new, imp_feat_new)

save(X_all, y, y_index, ind_drop, file="stacking_new.RData")


#generate stacking_old##########################

imp_var = as.matrix(importance_matrix)[1:100, 1]
imp_feat = df_all[, (names(df_all) %in% imp_var)]
imp_feat = imp_feat[rep(1:nrow(df_all), times=df_all$quantity), ]
imp_feat = as.matrix(imp_feat)
imp_feat_old = imp_feat

y_pred_prob_feat_xgb_baseline_old_ss95_cs75 = y_pred_prob_feat
y_pred_prob_feat_xgb_baseline_old_ss95_cs80 = y_pred_prob_feat
y_pred_prob_feat_xgb_baseline_old_ss95_cs85 = y_pred_prob_feat
y_pred_prob_feat_xgb_drop_gift_old_ss95_cs80 = y_pred_prob_feat
y_pred_prob_feat_xgb_drop_lowfreq_cust_old_ss95_cs80 = y_pred_prob_feat
y_pred_prob_feat_xgb_drop_feat_old_ss95_cs75 = y_pred_prob_feat
y_pred_prob_feat_xgb_drop_feat_old_ss95_cs80 = y_pred_prob_feat
y_pred_prob_feat_xgb_drop_feat_old_ss95_cs85 = y_pred_prob_feat
y_pred_prob_feat_xgb_likelihood_cust_old_ss95_cs80 = y_pred_prob_feat
y_pred_prob_feat_xgb_likelihood_cust_old_ss90_cs85 = y_pred_prob_feat
y_pred_prob_feat_xgb_likelihood_month_old_ss95_cs80 = y_pred_prob_feat
y_pred_prob_feat_xgb_old = c(y_pred_prob_feat_xgb_baseline_old_ss95_cs75, y_pred_prob_feat_xgb_baseline_old_ss95_cs80, y_pred_prob_feat_xgb_baseline_old_ss95_cs85, y_pred_prob_feat_xgb_drop_gift_old_ss95_cs80, y_pred_prob_feat_xgb_drop_lowfreq_cust_old_ss95_cs80, y_pred_prob_feat_xgb_drop_feat_old_ss95_cs75, y_pred_prob_feat_xgb_drop_feat_old_ss95_cs80, y_pred_prob_feat_xgb_drop_feat_old_ss95_cs85, y_pred_prob_feat_xgb_likelihood_cust_old_ss95_cs80, y_pred_prob_feat_xgb_likelihood_cust_old_ss90_cs85, y_pred_prob_feat_xgb_likelihood_month_old_ss95_cs80)

y_pred_prob_feat_dl_old = pred
y_pred_prob_feat_rf_old = rf_stack

X_all = cbind(y_pred_prob_feat_xgb_old, y_pred_prob_feat_dl_old, y_pred_prob_feat_rf_old, imp_feat_old)

save(X_all, y, y_index, ind_drop, file="stacking_old.RData")

