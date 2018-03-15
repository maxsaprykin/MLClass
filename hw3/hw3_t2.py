def f(x):
  if(x<0):
      if(x<-1.5):
          return y[X < -1.5].mean()
      else:
          return y[(X>=-1.5) & (X<0)].mean()
  else:
      if(x<1.5):
          return y[(X>=0) & (X<1.5)].mean()
      else:
          return y[X>=1.5].mean()
    
new_X = np.arange(-2, 2, 0.05).tolist()
new_Y = [f(i) for i in new_X]
plt.plot(new_X, new_Y);