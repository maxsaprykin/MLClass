rf = RandomForestClassifier(random_state=17)

tree_grid = GridSearchCV (rf, forest_params, cv=skf, scoring='roc_auc')
tree_grid.fit(X, y)

# 1 вопрос

print('Лучшее значение: ', tree_grid.best_score_)

# 2 вопрос
stability = tree_grid.cv_results_["std_test_score"][np.argmax(tree_grid.cv_results_["mean_test_score"])]*100
print('Устойчивость: ', stability)

tree_grid.best_params_