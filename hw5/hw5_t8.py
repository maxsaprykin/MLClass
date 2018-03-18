from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline


# будем разбивать на 3 фолда
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=17)

# в Pipeline будем сразу преобразовать наш текст и обучать логистическую регрессию
classifier = Pipeline([
    ('vectorizer', CountVectorizer(max_features = 100000, ngram_range = (1, 3))),
    ('clf', LogisticRegression(random_state=17))])
logit_pipe_params = {'clf__C': [0.1, 1, 10, 100]}

logit_grid = GridSearchCV (classifier, logit_pipe_params, cv=skf, scoring='roc_auc')
logit_grid.fit(X_text, y_text)

# 1 вопрос

print('Лучшее значение: ', logit_grid.best_score_)

# 2 вопрос
stability = logit_grid.cv_results_["std_test_score"][np.argmax(logit_grid.cv_results_["mean_test_score"])]*100
print('Устойчивость: ', stability)