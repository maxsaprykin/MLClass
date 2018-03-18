classifier = Pipeline([
    ('vectorizer', CountVectorizer(max_features=100000, ngram_range=(1, 3))),
    ('clf', RandomForestClassifier(random_state=17, n_jobs=-1))])

min_samples_leaf = [1, 2, 3]
max_features = [0.3, 0.5, 0.7]
max_depth = [None]

forest_params = {'clf__max_depth': max_depth,
               'clf__max_features': max_features_values,
                'clf__min_samples_leaf': min_samples_leaf}


forest_grid = GridSearchCV (classifier, forest_params, cv=skf, scoring='roc_auc')
forest_grid.fit(X_text, y_text)

# 1 вопрос

print('Лучшее значение: ', forest_grid.best_score_)

# 2 вопрос
stability = forest_grid.cv_results_["std_test_score"][np.argmax(forest_grid.cv_results_["mean_test_score"])]*100
print('Устойчивость: ', stability)