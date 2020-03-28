import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from scipy.optimize import curve_fit

colors = ['darkviolet','blue','green','gold','darkorange','red','gray','black']
region = np.array([0,1,1,1,2,1,3,2,1,1,1,1,1,1,3,1,3,3,1,4,0,1,2,0,0,2,1,2,1,3,3,0])
# 0: china, 1: europe, 2: asia, 3: US, 4: others

def linear_func(x,a,b):
	return a*x+b

def quadratic_func(x,a,b,c):
	return a*x*x+b*x+c

def plot_scatter_with_point_colors(xdata,ydata):
	xdata_np = np.array(xdata)
	ydata_np = np.array(ydata)
	for idx in range(1,2):
		plt.scatter(xdata_np[region==idx],ydata_np[region==idx],c=colors[idx])

def plot_dependence(func=linear_func):
	data = pd.read_csv('data\\model_param_results.csv')
	alpha = data.iloc[:,1]
	beta = data.iloc[:,2]
	alpha[21] = np.nan
	beta[21] = np.nan
	multiplier = data.iloc[:,6]
	temp = data.iloc[:,11]
	humidity = data.iloc[:,12]
	print(temp[25])
	temp[25] = 50
	humidity[25] = 0.78
	plt.figure()
	plt.subplot(221)
	# plt.scatter(temp,alpha)
	plot_scatter_with_point_colors(temp,alpha)
	# popt, pcov = curve_fit(func,temp,alpha)
	# plt.plot(temp,func(temp,*popt),'r-')
	plt.title('Temperature vs alpha')
	plt.subplot(222)
	# plt.scatter(temp,beta)
	plot_scatter_with_point_colors(temp,beta)
	plt.title('Temperature vs beta')
	plt.subplot(223)
	# plt.scatter(humidity,alpha)
	plot_scatter_with_point_colors(humidity,alpha)
	plt.title('Humidity vs alpha')
	plt.subplot(224)
	# plt.scatter(humidity,beta)
	plot_scatter_with_point_colors(humidity,beta)
	plt.title('Humidity vs beta')
	# plt.show()

def plot_dependence_corrected(func=linear_func):
	data = pd.read_csv('data\\model_param_results.csv')
	alpha = data.iloc[:,1]
	beta = data.iloc[:,2]
	alpha[21] = np.nan
	beta[21] = np.nan
	multiplier = data.iloc[:,6]
	alpha_corrected = multiplier*alpha
	beta_corrected = beta + alpha*(multiplier-1)
	temp = data.iloc[:,11]
	humidity = data.iloc[:,12]
	print(temp[25])
	temp[25] = 50
	humidity[25] = 0.78
	plt.figure()
	plt.subplot(221)
	# plt.scatter(temp,alpha_corrected)
	plot_scatter_with_point_colors(temp,alpha_corrected)
	# popt, pcov = curve_fit(func,temp,alpha_corrected)
	# plt.plot(temp,func(temp,*popt),'r-')
	plt.title('Temperature vs alpha (corrected)')
	plt.subplot(222)
	# plt.scatter(temp,beta_corrected)
	plot_scatter_with_point_colors(temp,beta_corrected)
	plt.title('Temperature vs beta (corrected)')
	plt.subplot(223)
	# plt.scatter(humidity,alpha_corrected)
	plot_scatter_with_point_colors(humidity,alpha_corrected)
	plt.title('Humidity vs alpha (corrected)')
	plt.subplot(224)
	# plt.scatter(humidity,beta_corrected)
	plot_scatter_with_point_colors(humidity,beta_corrected)
	plt.title('Humidity vs beta (corrected)')
	# plt.show()

if __name__ =='__main__':
	plot_dependence(func=quadratic_func)
	plot_dependence_corrected(func=quadratic_func)
	plt.show()