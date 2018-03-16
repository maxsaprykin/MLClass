from sklearn.ensemble import RandomForestClassifier
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)
#rf = RandomForestClassifierCustom(max_depth=7, max_features=6)
rf = RandomForestClassifier(n_jobs=1, random_state=17, max_depth=7, max_features=6)
cross_val_score(rf, X, y, cv=skf, scoring='roc_auc').mean()