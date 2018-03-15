df['age_in_years'] = df['age'] / 365.25
new_df = pd.get_dummies(df, columns = ['cholesterol', 'gluc'])
y = new_df['cardio']
new_df.drop(['cardio'], axis=1, inplace=True)
new_df.head()

X_train, X_valid, y_train, y_valid = train_test_split(new_df.values, y, test_size=0.3,
                                                          random_state=17)
														  
tree = DecisionTreeClassifier(max_depth=3, random_state=17)

%%time
tree.fit(X_train, y_train)

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
import pydotplus
from IPython.display import Image

export_graphviz(tree, out_file="tree.dot", feature_names=new_df.columns, filled=True)
graph = pydotplus.graphviz.graph_from_dot_file('tree.dot')
Image(graph.create_png())

														  