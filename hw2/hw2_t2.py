train_uniques = pd.melt(frame=train, value_vars=['height','gender'])

train_uniques = pd.DataFrame(train_uniques.groupby(['variable', 
                                                    'value'])['value'].count()) 
													
train.groupby('gender')[['height']].mean()

train_uniques = pd.melt(frame=train, value_vars=['height'], id_vars='gender')
sns.violinplot(x='variable', y='value', hue='gender', data=train_uniques)													