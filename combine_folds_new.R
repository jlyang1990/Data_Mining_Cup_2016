y_pred_prob_stack_xgb_new_fold_seed0_1 = y_pred_prob

y_pred_prob_stack_xgb_new_fold_seed0_2 = y_pred_prob

y_pred_prob_stack_xgb_new_fold_seed0 = y_pred_prob_stack_xgb_new_fold_seed0_1+
									   y_pred_prob_stack_xgb_new_fold_seed0_2

y_pred_prob_stack_xgb_new_fold_seed555_1 = y_pred_prob

y_pred_prob_stack_xgb_new_fold_seed555_2 = y_pred_prob

y_pred_prob_stack_xgb_new_fold_seed555 = y_pred_prob_stack_xgb_new_fold_seed555_1+
									   y_pred_prob_stack_xgb_new_fold_seed555_2

y_pred_prob_stack_xgb_new_fold_seed666_1 = y_pred_prob

y_pred_prob_stack_xgb_new_fold_seed666_2 = y_pred_prob

y_pred_prob_stack_xgb_new_fold_seed666 = y_pred_prob_stack_xgb_new_fold_seed666_1+
									   y_pred_prob_stack_xgb_new_fold_seed666_2

y_pred_prob_stack_xgb_new_fold_seed777_1 = y_pred_prob

y_pred_prob_stack_xgb_new_fold_seed777_2 = y_pred_prob

y_pred_prob_stack_xgb_new_fold_seed777 = y_pred_prob_stack_xgb_new_fold_seed777_1+
									   y_pred_prob_stack_xgb_new_fold_seed777_2

y_pred_prob_stack_xgb_new_fold_seed888_1 = y_pred_prob

y_pred_prob_stack_xgb_new_fold_seed888_2 = y_pred_prob

y_pred_prob_stack_xgb_new_fold_seed888 = y_pred_prob_stack_xgb_new_fold_seed888_1+
									   y_pred_prob_stack_xgb_new_fold_seed888_2

y_pred_prob_stack_xgb_new_fold_seed999_1 = y_pred_prob

y_pred_prob_stack_xgb_new_fold_seed999_2 = y_pred_prob

y_pred_prob_stack_xgb_new_fold_seed999 = y_pred_prob_stack_xgb_new_fold_seed999_1+
									   y_pred_prob_stack_xgb_new_fold_seed999_2

y_pred_prob_new = y_pred_prob_stack_xgb_new_fold_seed0_1+
			  y_pred_prob_stack_xgb_new_fold_seed0_2+
			  y_pred_prob_stack_xgb_new_fold_seed555_1+
			  y_pred_prob_stack_xgb_new_fold_seed555_2+
			  y_pred_prob_stack_xgb_new_fold_seed666_1+
			  y_pred_prob_stack_xgb_new_fold_seed666_2+
			  y_pred_prob_stack_xgb_new_fold_seed777_1+
			  y_pred_prob_stack_xgb_new_fold_seed777_2+
			  y_pred_prob_stack_xgb_new_fold_seed888_1+
			  y_pred_prob_stack_xgb_new_fold_seed888_2+
			  y_pred_prob_stack_xgb_new_fold_seed999_1+
			  y_pred_prob_stack_xgb_new_fold_seed999_2

y_pred_prob_new = y_pred_prob_new/6

save(y_pred_prob_new, file = "DMC16_stack_xgb_new.RData")