tree_grid = GridSearchCV (dt,tree_params,cv=skf,scoring='roc_auc')
tree_grid.fit(X, y)

# 1 вопрос
tree_grid.best_score_

# 2 вопрос
tree_grid.cv_results_["std_test_score"][np.argmax(tree_grid.cv_results_["mean_test_score"])]*100