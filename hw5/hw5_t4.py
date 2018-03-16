from sklearn.model_selection import cross_val_score

from sklearn.base import BaseEstimator

class RandomForestClassifierCustom(BaseEstimator):
    def __init__(self, n_estimators=10, max_depth=7, max_features=6, random_state=17):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        # в данном списке будем хранить отдельные деревья
        self.trees = []
        # тут будем хранить списки индексов признаков, на которых обучалось каждое дерево 
        self.feat_ids_by_tree = []
                
    def fit(self, X, y):
        for i in (0, self.n_estimators - 1):
            np.random.seed(self.random_state + i)            
            features = np.random.choice(X.columns, self.max_features, replace=False)
            self.feat_ids_by_tree.append(features)
            bootstrap_inds = np.random.choice(len(X), len(X), replace=True)
            new_X = X.iloc[bootstrap_inds][features]
            new_y = y.iloc[bootstrap_inds]
            dt = DecisionTreeClassifier(random_state = self.random_state, max_depth=self.max_depth, max_features=self.max_features, class_weight='balanced')
            dt.fit(new_X, new_y)
            self.trees.append(dt)
        return self    
    
    def predict_proba(self, X):
        probas = []
        for tree, features in zip(self.trees, self.feat_ids_by_tree):
            probas.append(tree.predict_proba(X[features]))
        return np.mean(probas, axis=0)
		
		
		
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)
rf = RandomForestClassifierCustom(max_depth=7, max_features=6)
cross_val_score(rf, X, y, cv=skf, scoring='roc_auc').mean()		
            