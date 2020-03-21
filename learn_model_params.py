import numpy as np
import matplotlib.pyplot as plt
from SEIR import corona_seir_model

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
	for idx,t in enumerate(T[1:]):
		update_mat = np.array([[(1-beta*E[-1]),(-beta*S[-1]),0],[(beta*E[-1]),(1+beta*S[-1]-alpha-gamma1),0],[0,alpha,(1-gamma2)]])
		add_mat = np.array([[0,(-S[-1]*E[-1]),0,0],[(-E[-1]),S[-1]*E[-1],(-E[-1]),0],[E[-1],0,0,(-I[-1])]])
		jacobian_mat = np.matmul(update_mat,jacobian_mat)+add_mat
		S1 = max(0,S[-1] - (beta*S[-1]*E[-1])*dt)
		E1 = E[-1] + (beta*S[-1]*E[-1] - alpha*E[-1] - gamma1*E[-1])*dt
		I1 = I[-1] + (alpha*E[-1] - gamma2*I[-1])*dt
		R1 = R[-1] + (gamma1*E[-1] + gamma2*I[-1])*dt
		S.append(S1)
		E.append(E1)
		I.append(I1)
		R.append(R1)
		loss.append((infected[idx+1]-I1)**2)
		alpha_update = lr*(infected[idx+1]-I1)*jacobian_mat[2,0]
		beta_update = lr*(infected[idx+1]-I1)*jacobian_mat[2,1]
		gamma1_update = lr*(infected[idx+1]-I1)*jacobian_mat[2,2]
		gamma2_update = lr*(infected[idx+1]-I1)*jacobian_mat[2,3]
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

if __name__=='__main__':
	T_max = 60
	dt = 1
	T = np.linspace(0,T_max,int(T_max/dt)+1)
	N = 10000
	init_exposed = 10
	init_vals = (N-init_exposed)/N,init_exposed/N,0,0
	alpha = 0.1
	beta = 2.0
	gamma1 = 0.25
	gamma2 = 0.01
	rho = 1.0
	# if model.__name__=='corona_seir_model':
	# 	params = alpha, beta*rho, gamma1, gamma2
	# elif model.__name__=='base_seir_model':
	# 	params = alpha, beta*rho, gamma1
	params = alpha, beta*rho, gamma1, gamma2
	sim_results = corona_seir_model(init_vals,params,T)
	plt.figure(1)
	plt.plot(T,sim_results[0],label='Susceptible')
	plt.plot(T,sim_results[1],label='Exposed')
	plt.plot(T,sim_results[2],label='Infected')
	plt.plot(T,sim_results[3],label='Recovered')
	plt.legend()
	# plt.show()
	# init_params = 0.1*alpha, beta*rho, gamma1, gamma2
	init_params = 0, 1, 0.2, 0
	loss_jacobian = epoch_fit_params_corona_seir(init_vals,init_params,T,sim_results[2],lr=0.02)
	plt.figure(2)
	plt.plot(T,loss_jacobian[0])
	plt.figure(3)
	# plt.plot(T,loss_jacobian[1],label='gamma2_S')
	# plt.plot(T,loss_jacobian[2],label='gamma2_E')
	# plt.plot(T,loss_jacobian[3],label='gamma2_I')
	plt.plot(T,loss_jacobian[1],label='alpha')
	plt.plot(T,loss_jacobian[2],label='beta')
	plt.plot(T,loss_jacobian[3],label='gamma1')
	plt.plot(T,loss_jacobian[4],label='gamma2')
	plt.legend()
	plt.show()

	total_epochs = 200
	lr = 0.02
	lrd = 0.04
	curr_params = 0.001,0.5,0.1,0.1
	loss_arr = []
	alpha_arr = []
	beta_arr = []
	gamma1_arr = []
	gamma2_arr = []
	for epoch in range(total_epochs):
		curr_lr = lr/(1+epoch*lrd)
		loss_jacobian = epoch_fit_params_corona_seir(init_vals,curr_params,T,sim_results[2],lr=curr_lr)
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
	plt.plot(list(range(total_epochs)),loss_arr)
	plt.figure(2)
	plt.plot(list(range(total_epochs)),alpha_arr,label='alpha')
	plt.plot(list(range(total_epochs)),beta_arr,label='beta')
	plt.plot(list(range(total_epochs)),gamma1_arr,label='gamma1')
	plt.plot(list(range(total_epochs)),gamma2_arr,label='gamma2')
	plt.legend()
	plt.show()