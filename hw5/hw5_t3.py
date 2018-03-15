tree = DecisionTreeClassifier(max_depth=3, random_state=17)
tree.fit(X, y)

tree_grid = GridSearchCV (tree,tree_params,cv=skf,scoring='roc_auc')
tree_grid.fit(X, y)

tree_grid.best_score_