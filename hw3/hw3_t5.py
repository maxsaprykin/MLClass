new_df = pd.get_dummies(df, columns = ['cholesterol'])
new_df['age_in_years'] = new_df['age'] / 365.25

d =  {True: 1, False: 0}
new_df['age_1'] = (new_df['age_in_years'] >= 45) & (new_df['age_in_years'] < 50)
new_df['age_1'] = new_df['age_1'].map(d)

new_df['age_2'] = (new_df['age_in_years'] >= 50) & (new_df['age_in_years'] < 55)
new_df['age_2'] = new_df['age_2'].map(d)

new_df['age_3'] = (new_df['age_in_years'] >= 55) & (new_df['age_in_years'] < 60)
new_df['age_3'] = new_df['age_3'].map(d)

new_df['age_4'] = (new_df['age_in_years'] >= 60) & (new_df['age_in_years'] < 65)
new_df['age_4'] = new_df['age_4'].map(d)

new_df['ap_hi_1'] = (new_df['ap_hi'] >= 120) & (new_df['ap_hi'] < 140)
new_df['ap_hi_1'] = new_df['ap_hi_1'].map(d)

new_df['ap_hi_2'] = (new_df['ap_hi'] >= 140) & (new_df['ap_hi'] < 160)
new_df['ap_hi_2'] = new_df['ap_hi_2'].map(d)

new_df['ap_hi_3'] = (new_df['ap_hi'] >= 160) & (new_df['ap_hi'] < 180)
new_df['ap_hi_3'] = new_df['ap_hi_3'].map(d)

d2 = {1: 0, 2: 1}
new_df['gender'] = new_df['gender'].map(d2)
new_df.rename(index=str, columns={"gender": "male"})

y = new_df['cardio']
new_df.drop(['cardio', 'age', 'height', 'weight','ap_hi', 'ap_lo', 'gluc', 'alco', 'active', 'age_in_years'], axis=1, inplace=True)
x = new_df.values

tree = DecisionTreeClassifier(max_depth=3, random_state=17)
tree.fit(new_df, y)

export_graphviz(tree, out_file="tree.dot", feature_names=new_df.columns, filled=True)
graph = pydotplus.graphviz.graph_from_dot_file('tree.dot')
Image(graph.create_png())