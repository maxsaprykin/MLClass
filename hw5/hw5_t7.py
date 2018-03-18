from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

scaler = StandardScaler()
logit = LogisticRegression(random_state=17, class_weight='balanced')

logit_pipe = Pipeline([('scaler', scaler), ('logit', logit)])
logit_pipe_params = {'logit__C': np.logspace(-8, 8, 17)}l

ogit_grid = GridSearchCV (logit_pipe, logit_pipe_params, cv=skf, scoring='roc_auc')
logit_grid.fit(X, y)

print('Лучшее значение на обучении: ', logit_grid.best_score_)
stability = logit_grid.cv_results_["std_test_score"][np.argmax(logit_grid.cv_results_["mean_test_score"])]*100
print('Устойчивость: ', stability)