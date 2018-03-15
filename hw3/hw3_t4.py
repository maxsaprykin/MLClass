tree_pred = tree.predict(X_valid)
accuracy_score(y_valid, tree_pred)

tree_params = {'max_depth': list(range(2, 11))}

tree_grid = GridSearchCV(tree, tree_params,
                         cv=5, n_jobs=-1,
                        verbose=True)

tree_grid.fit(X_train, y_train)

X = range(2,11)
Y = tree_grid.cv_results_['mean_test_score']
plt.plot(X, Y);

tree_grid.best_params_

accuracy_score(y_valid, tree_grid.predict(X_valid))