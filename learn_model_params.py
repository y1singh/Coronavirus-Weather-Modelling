import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from SEIR import corona_seir_model, base_seir_model

# define param update equations
# plot the partial derivative of each param with time
# update params with some lr and observe how they change over time compared to the true value

def epoch_fit_params_corona_seir(init_vals, init_params, T, infected, lr=1e-2):
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
	for idx,t in enumerate(T[1:]):
		# print("{} {} {} {} {:.3f} {:.3f} {:.3f} {:3.3f}".format(t,alpha,beta,gamma1,S[-1],E[-1],I[-1],np.max(jacobian_mat)))
		update_mat = np.array([[(1-beta*E[-1]),(-beta*S[-1]),0],[(beta*E[-1]),(1+beta*S[-1]-alpha-gamma1),0],[0,alpha,(1-gamma2)]])
		add_mat = np.array([[0,(-S[-1]*E[-1]),0,0],[(-E[-1]),S[-1]*E[-1],(-E[-1]),0],[E[-1],0,0,(-I[-1])]])
		jacobian_mat = np.matmul(update_mat,jacobian_mat)+add_mat
		S1 = S[-1] - (beta*S[-1]*E[-1])*dt
		E1 = E[-1] + (beta*S[-1]*E[-1] - alpha*E[-1] - gamma1*E[-1])*dt
		I1 = I[-1] + (alpha*E[-1] - gamma2*I[-1])*dt
		R1 = R[-1] + (gamma1*E[-1] + gamma2*I[-1])*dt
		S.append(S1)
		E.append(E1)
		I.append(I1)
		R.append(R1)
		# print(T[idx+1],infected[T[idx+1]])
		loss.append(((infected[T[idx+1]]-I1))**2)
		alpha_update = lr*(infected[T[idx+1]]-I1)*jacobian_mat[2,0]
		beta_update = lr*(infected[T[idx+1]]-I1)*jacobian_mat[2,1]
		gamma1_update = lr*(infected[T[idx+1]]-I1)*jacobian_mat[2,2]
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

def epoch_fit_params_corona_seir_log(init_vals, init_params, T, infected, lr=1e-2):
	infected[infected<0]=0
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
	for idx,t in enumerate(T[1:]):
		update_mat = np.array([[(1-beta*E[-1]),(-beta*S[-1]),0],[(beta*E[-1]),(1+beta*S[-1]-alpha-gamma1),0],[0,alpha,(1-gamma2)]])
		add_mat = np.array([[0,(-S[-1]*E[-1]),0,0],[(-E[-1]),S[-1]*E[-1],(-E[-1]),0],[E[-1],0,0,(-I[-1])]])
		jacobian_mat = np.matmul(update_mat,jacobian_mat)+add_mat
		min_val = 100
		S1 = S[-1] - (beta*S[-1]*E[-1])*dt
		if S1<min_val: min_val=S1
		E1 = E[-1] + (beta*S[-1]*E[-1] - alpha*E[-1] - gamma1*E[-1])*dt
		if E1<min_val: min_val=E1
		I1 = I[-1] + (alpha*E[-1] - gamma2*I[-1])*dt
		if I1<min_val: min_val=I1
		R1 = R[-1] + (gamma1*E[-1] + gamma2*I[-1])*dt
		if R1<min_val: min_val=R1
		N1 = S1+E1+I1+R1-4*min_val
		S1 = (S1-min_val)/N1
		E1 = (E1-min_val)/N1
		I1 = (I1-min_val)/N1
		R1 = (R1-min_val)/N1
		# print(S1,E1,I1,R1,S1+E1+I1+R1)
		S.append(S1)
		E.append(E1)
		I.append(I1)
		R.append(R1)
		# print(t,infected[T[idx+1]],np.log(1e-10+infected[T[idx+1]]),I1,np.log(1e-10+I1))
		loss.append((np.log(1e-10+infected[T[idx+1]])-np.log(1e-10+I1))**2)
		alpha_update = lr*((np.log(1e-10+infected[T[idx+1]])-np.log(1e-10+I1)))*jacobian_mat[2,0]
		beta_update = lr*((np.log(1e-10+infected[T[idx+1]])-np.log(1e-10+I1)))*jacobian_mat[2,1]
		gamma1_update = 0 #lr*((np.log(1e-10+infected[T[idx+1]])-np.log(1e-10+I1)))*jacobian_mat[2,2]
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

def test_fitting(model):
	T_max = 60
	dt = 1
	T = np.linspace(0,T_max,int(T_max/dt)+1).astype(int)
	N = 100000
	init_exposed = 10
	init_vals = (N-init_exposed)/N,init_exposed/N,0,0
	alpha = 0.4
	beta = 1.7
	gamma1 = 0.01
	gamma2 = 0.05
	rho = 1.0
	if model.__name__=='corona_seir_model':
		sim_params = alpha, beta*rho, gamma1, gamma2
	elif model.__name__=='base_seir_model':
		sim_params = alpha, beta*rho, 0, gamma2
	sim_results = corona_seir_model(init_vals,sim_params,T)
	noise = np.random.normal(0,0.0,size=(len(sim_results[2]),))
	total_epochs = 4000
	lr = 0.0004
	lrd = 0.0
	curr_params = 0.0,0.4,0,gamma2
	loss_arr = []
	alpha_arr = []
	beta_arr = []
	gamma1_arr = []
	gamma2_arr = []
	for epoch in range(total_epochs):
		curr_lr = lr/(1+epoch*lrd)
		loss_jacobian = epoch_fit_params_corona_seir_log(init_vals,curr_params,T,sim_results[2]+noise,lr=curr_lr)
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

	learned_results = corona_seir_model(init_vals,curr_params,T)
	# plt.figure(2)
	plt.subplot(212)
	p = plt.plot(T,sim_results[0],label='GT Susceptible')
	plt.plot(T,learned_results[0],color=p[0].get_color(),linestyle='--',label='Predicted Susceptible')
	p = plt.plot(T,sim_results[1],label='GT Exposed')
	plt.plot(T,learned_results[1],color=p[0].get_color(),linestyle='--',label='Predicted Exposed')
	p = plt.plot(T,sim_results[2],label='GT Infected')
	plt.plot(T,learned_results[2],color=p[0].get_color(),linestyle='--',label='Predicted Infected')
	p = plt.plot(T,sim_results[3],label='Recovered')
	plt.plot(T,learned_results[3],color=p[0].get_color(),linestyle='--',label='Predicted Recovered')
	plt.legend()
	plt.ylabel('Fraction of population')
	plt.xlabel('Time (days)')
	plt.title('Simulated (GT) and learned models')
	# plt.figure()
	# plt.plot(sim_results[2],label='GT infected')
	# plt.plot(sim_results[2]+noise,label='noisy infected')
	# plt.legend()
	plt.show()

def fit_to_data(model,gt_infected,population1,population2=None):
	if population2 is None:
		population2 = population1
	T_max = gt_infected.shape[0]-1
	dt = 1
	T = np.linspace(0,T_max,int(T_max/dt)+1).astype(int)
	N = population1
	N2 = population2
	init_exposed = int(gt_infected[0]*1)
	init_vals = (N-init_exposed-gt_infected[0])/N,init_exposed/N,gt_infected[0]/N,0
	gamma2 = 0.01
	rho = 1.0
	total_epochs = 4000
	# lr = 0.04
	# lrd = 0.01
	lr = 0.0003
	lrd = 0.0
	curr_params = 0.4,0.5,0,gamma2
	loss_arr = []
	alpha_arr = []
	beta_arr = []
	gamma1_arr = []
	gamma2_arr = []
	for epoch in range(total_epochs):
		curr_lr = lr/(1+epoch*lrd)
		loss_jacobian = epoch_fit_params_corona_seir_log(init_vals,curr_params,T,gt_infected/N,lr=curr_lr)
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
	# plt.plot(list(range(total_epochs)),gamma2_arr,label='gamma2')
	plt.title('Learning trajectory')
	plt.ylabel('Parameter value')
	plt.xlabel('Epochs')
	plt.legend()
	# print(init_vals)
	T_pred1 = np.linspace(0,T_max,int(T_max/dt)+1).astype(int)
	learned_results1 = corona_seir_model(init_vals,curr_params,T_pred1)
	# init_vals2 = (N/N2)*learned_results1[:,-1]
	# init_vals2[0] = 1 - (init_vals2[1]+init_vals2[2]+init_vals2[3])
	init_vals2 = (N2-init_exposed-gt_infected[0])/N2,init_exposed/N2,gt_infected[0]/N2,0
	print(learned_results1[:,-1],init_vals2)
	T_pred2 = np.linspace(0,2*T_max,int(2*T_max/dt)+1).astype(int)
	learned_results2 = corona_seir_model(init_vals2,curr_params,T_pred2)
	print(learned_results1[0][-1],learned_results1[1][-1],learned_results1[2][-1],learned_results1[3][-1],learned_results1[0][-1]*N,learned_results2[0][0]*N2)
	# plt.figure(2)
	plt.subplot(212)
	# p = plt.plot(T,sim_results[0],label='GT Susceptible')
	p = plt.plot(T_pred1,learned_results1[0],linestyle='--',label='Predicted Susceptible')
	plt.plot(T_pred2,learned_results2[0],color=p[0].get_color(),linestyle='--')
	# p = plt.plot(T,sim_results[1],label='GT Exposed')
	p = plt.plot(T_pred1,(N/N2)*learned_results1[1],linestyle='--',label='Predicted Exposed')
	plt.plot(T_pred2,learned_results2[1],color=p[0].get_color(),linestyle='--')
	p = plt.plot(gt_infected[T]/N2,label='GT Infected')
	plt.plot(T_pred1,(N/N2)*learned_results1[2],color=p[0].get_color(),linestyle='--',label='Predicted Infected')
	plt.plot(T_pred2,learned_results2[2],color=p[0].get_color(),linestyle='--')
	# p = plt.plot(T,sim_results[3],label='Recovered')
	p = plt.plot(T_pred1,learned_results1[3],linestyle='--',label='Predicted Recovered')
	plt.plot(T_pred2,learned_results2[3],color=p[0].get_color(),linestyle='--')
	plt.legend()
	plt.ylabel('Fraction of population')
	plt.xlabel('Time (days)')
	plt.title('GT and learned models')
	plt.show()

if __name__=='__main__':
	# test_fitting(corona_seir_model)
	data = pd.read_csv('data\\time_series_covid_19_confirmed.csv')
	gt_infected = np.array(data.iloc[16,35:]).astype(int)
	plt.plot(gt_infected); plt.show()
	fit_to_data(corona_seir_model,gt_infected,1e6,1e6)
	# T_max = 60
	# dt = 1
	# T = np.linspace(0,T_max,int(T_max/dt)+1)
	# N = 10000
	# init_exposed = 10
	# init_vals = (N-init_exposed)/N,init_exposed/N,0,0
	# alpha = 0.2
	# beta = 1.5
	# gamma1 = 0.01
	# gamma2 = 0.2
	# rho = 1.0
	# # if model.__name__=='corona_seir_model':
	# # 	params = alpha, beta*rho, gamma1, gamma2
	# # elif model.__name__=='base_seir_model':
	# # 	params = alpha, beta*rho, gamma1
	# sim_params = alpha, beta*rho, gamma1, gamma2
	# sim_results = corona_seir_model(init_vals,sim_params,T)
	# plt.figure(1)
	# plt.plot(T,sim_results[0],label='Susceptible')
	# plt.plot(T,sim_results[1],label='Exposed')
	# plt.plot(T,sim_results[2],label='Infected')
	# plt.plot(T,sim_results[3],label='Recovered')
	# plt.legend()
	# # plt.show()
	# # init_params = 0.1*alpha, beta*rho, gamma1, gamma2
	# init_params = 0, 1.5, 0.2, 0
	# loss_jacobian = epoch_fit_params_corona_seir(init_vals,init_params,T,sim_results[2],lr=0.02)
	# plt.figure(2)
	# plt.plot(T,loss_jacobian[0])
	# plt.figure(3)
	# # plt.plot(T,loss_jacobian[1],label='gamma2_S')
	# # plt.plot(T,loss_jacobian[2],label='gamma2_E')
	# # plt.plot(T,loss_jacobian[3],label='gamma2_I')
	# plt.plot(T,loss_jacobian[1],label='alpha')
	# plt.plot(T,loss_jacobian[2],label='beta')
	# plt.plot(T,loss_jacobian[3],label='gamma1')
	# plt.plot(T,loss_jacobian[4],label='gamma2')
	# plt.legend()
	# plt.show()

	# total_epochs = 400
	# lr = 0.02
	# lrd = 0.04
	# # curr_params = 0.001,0.5,0.1,0.1
	# curr_params = 0,1,0,0.1
	# loss_arr = []
	# alpha_arr = []
	# beta_arr = []
	# gamma1_arr = []
	# gamma2_arr = []
	# for epoch in range(total_epochs):
	# 	curr_lr = lr/(1+epoch*lrd)
	# 	loss_jacobian = epoch_fit_params_corona_seir(init_vals,curr_params,T,sim_results[2],lr=curr_lr)
	# 	loss_epoch = np.sum(loss_jacobian[0])
	# 	new_alpha = max(0,curr_params[0]+np.sum(loss_jacobian[1]))
	# 	new_beta = max(0,curr_params[1]+np.sum(loss_jacobian[2]))
	# 	new_gamma1 = max(0,curr_params[2]+np.sum(loss_jacobian[3]))
	# 	new_gamma2 = max(0,curr_params[3]+np.sum(loss_jacobian[4]))
	# 	curr_params = new_alpha,new_beta,new_gamma1,new_gamma2
	# 	loss_arr.append(loss_epoch)
	# 	alpha_arr.append(new_alpha)
	# 	beta_arr.append(new_beta)
	# 	gamma1_arr.append(new_gamma1)
	# 	gamma2_arr.append(new_gamma2)
	# plt.figure()
	# plt.subplot(121)
	# plt.plot(list(range(total_epochs)),loss_arr)
	# plt.ylabel('Total MSE loss')
	# plt.xlabel('Epochs')
	# plt.subplot(122)
	# plt.plot(list(range(total_epochs)),alpha_arr,label='alpha')
	# plt.plot(list(range(total_epochs)),beta_arr,label='beta')
	# plt.plot(list(range(total_epochs)),gamma1_arr,label='gamma1')
	# plt.plot(list(range(total_epochs)),gamma2_arr,label='gamma2')
	# plt.title('Original values: $\\alpha$={},$\\beta$={},$\gamma_1$={},$\gamma_2$={}'.format(sim_params[0],sim_params[1],sim_params[2],sim_params[3]))
	# plt.ylabel('Parameter value')
	# plt.xlabel('Epochs')
	# plt.legend()
	# plt.show()