import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from scipy.optimize import curve_fit

colors = ['darkviolet','blue','green','gold','darkorange','red','gray','black']
region = np.array([0,1,1,1,2,1,3,2,1,1,1,1,1,1,3,1,3,3,1,4,0,1,2,0,0,2,1,2,1,3,3,0])
# 0: china, 1: europe, 2: asia, 3: US, 4: others
outliers = [0,4,6]
region = np.delete(region,outliers)

def linear_func(x,a,b):
	return a*x+b

def quadratic_func(x,a,b,c):
	return a*x*x+b*x+c

def cubic_func(x,a,b,c,d):
	return a*x*x*x+b*x*x+c*x+d	

def biquadratic_func(x,a,b,c,d,e):
	return a*x*x*x*x+b*x*x*x+c*x*x+d*x+e

def plot_scatter_with_point_colors(xdata,ydata):
	xdata_np = np.array(xdata)
	ydata_np = np.array(ydata)
	for idx in range(5):
		plt.scatter(xdata_np[region==idx],ydata_np[region==idx],c=colors[idx])

def plot_dependence(func=linear_func):
	data = pd.read_csv('data\\model_param_results.csv')
	data = data.drop(outliers,axis=0)
	alpha = data.iloc[:,1]
	beta = data.iloc[:,2]
	multiplier = data.iloc[:,6]
	temp = data.iloc[:,11]
	humidity = data.iloc[:,12]
	plt.figure()
	plt.subplot(221)
	# plt.scatter(temp,alpha)
	plot_scatter_with_point_colors(temp,alpha)
	# popt, pcov = curve_fit(func,temp,alpha)
	# plt.plot(temp,func(temp,*popt),'r-')
	plt.title('Temperature vs alpha [{:.3f}]'.format(np.corrcoef(temp,alpha)[0,1]))
	plt.subplot(222)
	# plt.scatter(temp,beta)
	plot_scatter_with_point_colors(temp,beta)
	plt.title('Temperature vs beta [{:.3f}]'.format(np.corrcoef(temp,beta)[0,1]))
	plt.subplot(223)
	# plt.scatter(humidity,alpha)
	plot_scatter_with_point_colors(humidity,alpha)
	plt.title('Humidity vs alpha [{:.3f}]'.format(np.corrcoef(humidity,alpha)[0,1]))
	plt.subplot(224)
	# plt.scatter(humidity,beta)
	plot_scatter_with_point_colors(humidity,beta)
	plt.title('Humidity vs beta [{:.3f}]'.format(np.corrcoef(humidity,beta)[0,1]))
	# plt.show()

def plot_dependence_corrected(func=linear_func):
	data = pd.read_csv('data\\model_param_results.csv')
	alpha = data.iloc[:,1]
	beta = data.iloc[:,2]
	multiplier = data.iloc[:,6]
	alpha_corrected = multiplier*alpha
	beta_corrected = beta + alpha*(multiplier-1)
	temp = data.iloc[:,11]
	humidity = data.iloc[:,12]
	plt.figure()
	plt.subplot(221)
	# plt.scatter(temp,alpha_corrected)
	plot_scatter_with_point_colors(temp,alpha_corrected)
	# popt, pcov = curve_fit(func,temp,alpha_corrected)
	# plt.plot(temp,func(temp,*popt),'r-')
	plt.title('Temperature vs alpha (corrected) [{:.3f}]'.format(np.corrcoef(temp,alpha_corrected)[0,1]))
	plt.subplot(222)
	# plt.scatter(temp,beta_corrected)
	plot_scatter_with_point_colors(temp,beta_corrected)
	plt.title('Temperature vs beta (corrected) [{:.3f}]'.format(np.corrcoef(temp,beta_corrected)[0,1]))
	plt.subplot(223)
	# plt.scatter(humidity,alpha_corrected)
	plot_scatter_with_point_colors(humidity,alpha_corrected)
	plt.title('Humidity vs alpha (corrected) [{:.3f}]'.format(np.corrcoef(humidity,alpha_corrected)[0,1]))
	plt.subplot(224)
	# plt.scatter(humidity,beta_corrected)
	plot_scatter_with_point_colors(humidity,beta_corrected)
	plt.title('Humidity vs beta (corrected) [{:.3f}]'.format(np.corrcoef(humidity,beta_corrected)[0,1]))
	# plt.show()

def plot_dependence_corrected2(funcT=linear_func,funcH=linear_func):
	data = pd.read_csv('data\\model_param_results.csv')
	data = data.drop(outliers,axis=0)
	alpha = data.iloc[:,1]
	beta = data.iloc[:,2]
	E0 = data.iloc[:,13]
	correct_E0 = 100
	alpha_corrected = (E0/correct_E0)*alpha
	beta_corrected = beta + alpha*((E0/correct_E0)-1)
	temp = data.iloc[:,11]
	humidity = 100*data.iloc[:,12]
	plt.figure()
	plt.subplot(221)
	# plt.scatter(temp,alpha_corrected)
	plot_scatter_with_point_colors(temp,alpha_corrected)
	plt.xlabel('Temperature ($^o$F)',size=12)
	plt.ylabel('$\\alpha$',size=16)
	popt, pcov = curve_fit(funcT,temp,alpha_corrected)
	temp_range = np.linspace(25,75,51)
	plt.plot(temp_range,funcT(temp_range,*popt),'r--')
	plt.title('Temperature vs alpha (corrected) [$\\rho$={:.3f}]'.format(np.corrcoef(temp,alpha_corrected)[0,1]),fontsize=14)
	plt.subplot(222)
	# plt.scatter(temp,beta_corrected)
	plot_scatter_with_point_colors(temp,beta_corrected)
	plt.xlabel('Temperature ($^o$F)',size=12)
	plt.ylabel('$\\beta$',size=16)
	popt, pcov = curve_fit(funcT,temp,alpha_corrected)
	temp_range = np.linspace(25,75,51)
	plt.plot(temp_range,funcT(temp_range,*popt),'r--')
	plt.title('Temperature vs beta (corrected) [$\\rho$={:.3f}]'.format(np.corrcoef(temp,beta_corrected)[0,1]),fontsize=14)
	plt.subplot(223)
	# plt.scatter(humidity,alpha_corrected)
	plot_scatter_with_point_colors(humidity,alpha_corrected)
	plt.xlabel('Humidity (%)',size=12)
	plt.ylabel('$\\alpha$',size=16)
	popt, pcov = curve_fit(funcH,humidity,alpha_corrected)
	humidity_range = np.linspace(35,100,10)
	plt.plot(humidity_range,funcH(humidity_range,*popt),'r--')
	plt.title('Humidity vs alpha (corrected) [$\\rho$={:.3f}]'.format(np.corrcoef(humidity,alpha_corrected)[0,1]),fontsize=14)
	plt.subplot(224)
	# plt.scatter(humidity,beta_corrected)
	plot_scatter_with_point_colors(humidity,beta_corrected)
	plt.xlabel('Humidity (%)',size=12)
	plt.ylabel('$\\beta$',size=16)
	popt, pcov = curve_fit(funcH,humidity,beta_corrected)
	humidity_range = np.linspace(35,100,10)
	plt.plot(humidity_range,funcH(humidity_range,*popt),'r--')
	plt.title('Humidity vs beta (corrected) [$\\rho$={:.3f}]'.format(np.corrcoef(humidity,beta_corrected)[0,1]),fontsize=14)
	# plt.show()

def permutation_tests(xdata,ydata,xlabel='x',ylabel='y',repeats=10000,plot=False):
	true_corrcoef = np.corrcoef(xdata,ydata)[0,1] 	# off-diagonal element
	null_dist = []
	for r in range(repeats):
		ydata_f = np.random.permutation(ydata)
		null_dist.append(np.corrcoef(xdata,ydata_f)[0,1]) 	# off-diagonal element
	null_dist = np.array(null_dist)
	# null_dist = np.sort(null_dist)
	if plot:
		plt.figure()
		plt.hist(null_dist,bins=50)
		plt.axvline(x=true_corrcoef)
	pval = np.mean(true_corrcoef<=null_dist)
	if pval>0.5: pval = 1-pval
	print("corrcoef({},{}) p-value: {:.5f}".format(xlabel,ylabel,pval))
	# plt.show()

def do_permutation_tests():
	data = pd.read_csv('data\\model_param_results.csv')
	data = data.drop(outliers,axis=0)
	alpha = data.iloc[:,1]
	beta = data.iloc[:,2]
	# multiplier = data.iloc[:,6]
	# alpha_corrected = multiplier*alpha
	# beta_corrected = beta + alpha*(multiplier-1)
	E0 = data.iloc[:,13]
	correct_E0 = 100
	alpha_corrected = (E0/correct_E0)*alpha
	beta_corrected = beta + alpha*((E0/correct_E0)-1)
	temp = data.iloc[:,11]
	humidity = data.iloc[:,12]
	# permutation_tests(temp,alpha,'temp','alpha')
	# permutation_tests(temp,beta,'temp','beta')
	# permutation_tests(humidity,alpha,'humidity','alpha')
	# permutation_tests(humidity,beta,'humidity','beta')
	permutation_tests(temp,alpha_corrected,'temp','alpha_corrected')
	permutation_tests(temp,beta_corrected,'temp','beta_corrected')
	permutation_tests(humidity,alpha_corrected,'humidity','alpha_corrected')
	permutation_tests(humidity,beta_corrected,'humidity','beta_corrected')

if __name__ =='__main__':
	plot_dependence(func=quadratic_func)
	plot_dependence_corrected2(funcT=cubic_func,funcH=linear_func)
	# do_permutation_tests()
	plt.show()