new_x = np.arange(-2, 2, 0.05)
new_y = np.array([y.mean() for i in new_x])
plt.plot(new_x, new_y);


new_x2 = np.arange(-2, 2, 0.05)
new_y2 = [y[X < 0].mean() for i in new_x2[new_x2 < 0]] + [y[X >= 0].mean() for i in new_x2[new_x2 >= 0]]
plt.plot(new_x2, new_y2);


def regression_var_criterion(X, y, t):
    return np.var(y)-((len(X[X<t])/len(X))*np.var(y[:len(X[X<t])]))-((len(X[X>=t])/len(X))*np.var(y[len(X[X<t]):]))

	
from pylab import rcParams
rcParams['figure.figsize'] = 18, 15

ts = np.arange(-1.9, 1.9, 0.05)
trr = [regression_var_criterion(X, y, i) for i in ts]
plt.plot(ts, trr);	