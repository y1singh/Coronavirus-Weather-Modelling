import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from SEIR import corona_seir_model_population

def epoch_fit_params_corona_seir_population(init_vals, init_params, T, infected, lr=1e-2):
	max_infected = np.max(infected)
	S0,E0,I0,R0 = init_vals
	N = S0+E0+I0+R0
	S, E, I , R = [S0], [E0], [I0], [R0]
	loss = [0]
	alpha, beta, gamma1, gamma2 = init_params
	dt = T[1]-T[0]
	jacobian_mat = np.zeros((3,4))
	# updated_alpha, updated_beta, updated_gamma1, updated_gamma2 = [alpha],[beta],[gamma1],[gamma2]
	update_alpha, update_beta, update_gamma1, update_gamma2 = [0],[0],[0],[0]
	# print(infected)
	for idx,t in enumerate(T[1:-3]):
		# print("{} {} {} {} {:.3f} {:.3f} {:.3f} {:3.3f}".format(t,alpha,beta,gamma1,S[-1],E[-1],I[-1],np.max(jacobian_mat)))
		update_mat = np.array([[(1-beta*E[-1]/N),(-beta*S[-1]/N),0],[(beta*E[-1]/N),(1+beta*S[-1]/N-alpha-gamma1),0],[0,alpha,(1-gamma2)]])
		add_mat = np.array([[0,(-S[-1]*(E[-1]/N)),0,0],[(-E[-1]),S[-1]*(E[-1]/N),(-E[-1]),0],[E[-1],0,0,(-I[-1])]])/max_infected
		jacobian_mat = np.matmul(update_mat,jacobian_mat)+add_mat
		min_val = N
		S1 = S[-1] - (beta*S[-1]*E[-1]/N)*dt
		if S1<min_val: min_val=S1
		E1 = E[-1] + (beta*S[-1]*E[-1]/N - alpha*E[-1] - gamma1*E[-1])*dt
		if E1<min_val: min_val=E1
		I1 = I[-1] + (alpha*E[-1] - gamma2*I[-1])*dt
		if I1<min_val: min_val=I1
		R1 = R[-1] + (gamma1*E[-1] + gamma2*I[-1])*dt
		if R1<min_val: min_val=R1
		N1 = S1+E1+I1+R1-4*min_val
		S1 = N*(S1-min_val)/N1
		E1 = N*(E1-min_val)/N1
		I1 = N*(I1-min_val)/N1
		R1 = N*(R1-min_val)/N1
		S.append(S1)
		E.append(E1)
		I.append(I1)
		R.append(R1)
		# print(T[idx+1],infected[T[idx+1]])
		loss.append(((infected[T[idx+1]]-I1)**2)**0.5)
		alpha_update = lr*(infected[T[idx+1]]-I1)*jacobian_mat[2,0]
		beta_update = lr*(infected[T[idx+1]]-I1)*jacobian_mat[2,1]
		gamma1_update = 0 #lr*(infected[T[idx+1]]-I1)*jacobian_mat[2,2]
		gamma2_update = 0 #lr*(infected[idx+1]-I1)*jacobian_mat[2,3]
		alpha_new = alpha+alpha_update
		beta_new = beta+beta_update
		gamma1_new = gamma1+gamma1_update
		gamma2_new = gamma2+gamma2_update
		update_alpha.append(alpha_update)
		update_beta.append(beta_update)
		update_gamma1.append(gamma1_update)
		update_gamma2.append(gamma2_update)
	# return np.stack([loss,alpha_vals_S,alpha_vals_E,alpha_vals_I])
	return np.stack([loss,update_alpha,update_beta,update_gamma1,update_gamma2])

def test_fitting_population(model):
	T_max = 60
	dt = 1
	T = np.linspace(0,T_max,int(T_max/dt)+1).astype(int)
	N = 1e7
	init_exposed = 10
	init_vals = (N-init_exposed),init_exposed,0,0
	alpha = 0.1
	beta = 0.5
	gamma1 = 0.01
	gamma2 = 0.05
	rho = 1.0
	sim_params = alpha, beta*rho, gamma1, gamma2
	sim_results = model(init_vals,sim_params,T)
	max_infected = np.max(sim_results[2])
	noise = np.random.normal(0,0.0,size=(len(sim_results[2]),))
	total_epochs = 4000
	lr = 0.0004/max_infected
	lrd = 0.0
	curr_params = 0.1,1.0,0.0,gamma2
	loss_arr = []
	alpha_arr = []
	beta_arr = []
	gamma1_arr = []
	gamma2_arr = []
	for epoch in range(total_epochs):
		curr_lr = lr/(1+epoch*lrd)
		loss_jacobian = epoch_fit_params_corona_seir_population(init_vals,curr_params,T,sim_results[2]+noise,lr=curr_lr)
		# breakpoint()
		loss_epoch = np.sum(loss_jacobian[0])
		new_alpha = max(0,curr_params[0]+np.sum(loss_jacobian[1]))
		new_beta = max(0,curr_params[1]+np.sum(loss_jacobian[2]))
		new_gamma1 = max(0,curr_params[2]+np.sum(loss_jacobian[3]))
		new_gamma2 = max(0,curr_params[3]+np.sum(loss_jacobian[4]))
		curr_params = new_alpha,new_beta,new_gamma1,new_gamma2
		loss_arr.append(loss_epoch)
		alpha_arr.append(new_alpha)
		beta_arr.append(new_beta)
		gamma1_arr.append(new_gamma1)
		gamma2_arr.append(new_gamma2)
	plt.figure(1)
	plt.subplot(221)
	plt.plot(list(range(total_epochs)),loss_arr)
	plt.ylabel('Total MSE loss')
	plt.xlabel('Epochs')
	plt.subplot(222)
	plt.plot(list(range(total_epochs)),alpha_arr,label='alpha')
	plt.plot(list(range(total_epochs)),beta_arr,label='beta')
	plt.plot(list(range(total_epochs)),gamma1_arr,label='gamma1')
	plt.plot(list(range(total_epochs)),gamma2_arr,label='gamma2')
	plt.title('Original values: $\\alpha$={},$\\beta$={},$\gamma_1$={},$\gamma_2$={}'.format(sim_params[0],sim_params[1],sim_params[2],sim_params[3]))
	plt.ylabel('Parameter value')
	plt.xlabel('Epochs')
	plt.legend()

	learned_results = model(init_vals,curr_params,T)
	# plt.figure(2)
	plt.subplot(212)
	p = plt.plot(T,sim_results[0]/N,label='GT Susceptible')
	plt.plot(T,learned_results[0]/N,color=p[0].get_color(),linestyle='--',label='Predicted Susceptible')
	p = plt.plot(T,sim_results[1]/N,label='GT Exposed')
	plt.plot(T,learned_results[1]/N,color=p[0].get_color(),linestyle='--',label='Predicted Exposed')
	p = plt.plot(T,sim_results[2]/N,label='GT Infected')
	plt.plot(T,learned_results[2]/N,color=p[0].get_color(),linestyle='--',label='Predicted Infected')
	p = plt.plot(T,sim_results[3]/N,label='Recovered')
	plt.plot(T,learned_results[3]/N,color=p[0].get_color(),linestyle='--',label='Predicted Recovered')
	plt.legend()
	plt.ylabel('Fraction of population')
	plt.xlabel('Time (days)')
	plt.title('Simulated (GT) and learned models')
	# plt.figure()
	# plt.plot(sim_results[2],label='GT infected')
	# plt.plot(sim_results[2]+noise,label='noisy infected')
	# plt.legend()
	plt.show()

def fit_to_data_population(model,gt_infected,population):
	T_max = gt_infected.shape[0]-1
	max_infected = np.max(gt_infected)
	dt = 1
	T = np.linspace(0,T_max,int(T_max/dt)+1).astype(int)
	N = population
	init_exposed = int(gt_infected[0]*2)
	init_vals = (N-init_exposed-gt_infected[0]),init_exposed,gt_infected[0],0
	gamma2 = 0.03
	rho = 1.0
	total_epochs = 40000
	# lr = 0.04
	# lrd = 0.01
	lr = 0.0002/max_infected
	lrd = 0.001
	curr_params = 0.2,0.5,0.0,gamma2
	loss_arr = []
	alpha_arr = []
	beta_arr = []
	gamma1_arr = []
	gamma2_arr = []
	for epoch in tqdm(range(total_epochs)):
		curr_lr = lr/(1+epoch*lrd)
		loss_jacobian = epoch_fit_params_corona_seir_population(init_vals,curr_params,T,gt_infected,lr=curr_lr)
		loss_epoch = np.sum(loss_jacobian[0])
		new_alpha = max(0,curr_params[0]+np.sum(loss_jacobian[1]))
		new_beta = max(0,curr_params[1]+np.sum(loss_jacobian[2]))
		new_gamma1 = max(0,curr_params[2]+np.sum(loss_jacobian[3]))
		new_gamma2 = max(0,curr_params[3]+np.sum(loss_jacobian[4]))
		curr_params = new_alpha,new_beta,new_gamma1,new_gamma2
		loss_arr.append(loss_epoch)
		alpha_arr.append(new_alpha)
		beta_arr.append(new_beta)
		gamma1_arr.append(new_gamma1)
		gamma2_arr.append(new_gamma2)
	best_epoch = np.argmin(np.array(loss_arr))
	plt.figure(1)
	plt.subplot(221)
	plt.axvline(x=best_epoch,color='k',linestyle='--')
	plt.plot(list(range(total_epochs)),loss_arr)
	plt.ylabel('Total MSE loss')
	plt.xlabel('Epochs')
	plt.subplot(222)
	plt.axvline(x=best_epoch,color='k',linestyle='--')
	plt.plot(list(range(total_epochs)),alpha_arr,label='alpha')
	plt.plot(list(range(total_epochs)),beta_arr,label='beta')
	plt.plot(list(range(total_epochs)),gamma1_arr,label='gamma1')
	# plt.plot(list(range(total_epochs)),gamma2_arr,label='gamma2')
	plt.title('Learning trajectory')
	plt.ylabel('Parameter value')
	plt.xlabel('Epochs')
	plt.legend()
	# print(init_vals)
	best_params = alpha_arr[best_epoch],beta_arr[best_epoch],gamma1_arr[best_epoch],gamma2
	T_pred = np.linspace(0,10+T_max,int((10+T_max)/dt)+1).astype(int)
	learned_results = model(init_vals,best_params,T_pred)
	# plt.figure(2)
	plt.subplot(212)
	# p = plt.plot(T,sim_results[0],label='GT Susceptible')
	# p = plt.plot(T_pred,learned_results[0]/N,linestyle='--',label='Predicted Susceptible')
	# p = plt.plot(T,sim_results[1],label='GT Exposed')
	# p = plt.plot(T_pred,learned_results[1]/N,linestyle='--',label='Predicted Exposed')
	p = plt.plot(gt_infected[T]/N,label='GT Infected')
	plt.plot(T_pred,learned_results[2]/N,color=p[0].get_color(),linestyle='--',label='Predicted Infected')
	# p = plt.plot(T,sim_results[3],label='Recovered')
	p = plt.plot(T_pred,learned_results[3]/N,linestyle='--',label='Predicted Recovered')
	plt.legend()
	plt.ylabel('Fraction of population')
	plt.xlabel('Time (days)')
	plt.title('GT and learned models')
	print("Best learned params: {} {} {} {}".format(alpha_arr[best_epoch],beta_arr[best_epoch],gamma1_arr[best_epoch],100*np.abs(learned_results[2,T_max]-gt_infected[-1])/gt_infected[-1]))
	print(learned_results[2,T_max],gt_infected[-1])
	plt.show()

if __name__=='__main__':
	# test_fitting_population(corona_seir_model_population)
	region_idx = 502
	region_population = 10045029
	time_series_start = 58
	time_series_end = 69
	data = pd.read_csv('data\\time_series_covid_19_confirmed.csv')
	gt_infected = np.array(data.iloc[region_idx,time_series_start:time_series_end]).astype(int)
	plt.plot(gt_infected); plt.show()
	fit_to_data_population(corona_seir_model_population,gt_infected,region_population)