import numpy as np
import matplotlib.pyplot as plt

def base_seir_model(init_vals,params,T):
	S0,E0,I0,R0 = init_vals
	N = S0+E0+I0+R0
	S, E, I , R = [S0], [E0], [I0], [R0]
	alpha, beta, gamma = params
	dt = T[1]-T[0]
	for t in T[1:]:
		S1 = S[-1] - (beta*S[-1]*E[-1])*dt
		E1 = E[-1] + (beta*S[-1]*E[-1] - alpha*E[-1])*dt
		I1 = I[-1] + (alpha*E[-1] - gamma*I[-1])*dt
		R1 = R[-1] + (gamma*I[-1])*dt
		S.append(S1)
		E.append(E1)
		I.append(I1)
		R.append(R1)
	return np.stack([S,E,I,R])

def corona_seir_model(init_vals,params,T):
	S0,E0,I0,R0 = init_vals
	N = S0+E0+I0+R0
	S, E, I , R = [S0], [E0], [I0], [R0]
	alpha, beta, gamma1, gamma2 = params
	dt = T[1]-T[0]
	for t in T[1:]:
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
		S.append(S1)
		E.append(E1)
		I.append(I1)
		R.append(R1)
	return np.stack([S,E,I,R])

def corona_seir_model_population(init_vals,params,T):
	S0,E0,I0,R0 = init_vals
	N = S0+E0+I0+R0
	S, E, I , R = [S0], [E0], [I0], [R0]
	alpha, beta, gamma1, gamma2 = params
	dt = T[1]-T[0]
	for t in T[1:]:
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
	return np.stack([S,E,I,R])

def simulate_model(model):		
	T_max = 60
	dt = 1
	T = np.linspace(0,T_max,int(T_max/dt)+1)
	N = 10000
	init_exposed = 10
	init_vals = (N-init_exposed),init_exposed,0,0
	alpha = 0.1
	beta = 1.5
	gamma1 = 0.25
	gamma2 = 0.01
	rho = 0.75
	if model.__name__=='corona_seir_model':
		params = alpha, beta*rho, gamma1, gamma2
	elif model.__name__=='base_seir_model':
		params = alpha, beta*rho, gamma1
	results = model(init_vals,params,T)
	plt.plot(T,results[0],label='Susceptible')
	plt.plot(T,results[1],label='Exposed')
	plt.plot(T,results[2],label='Infected')
	plt.plot(T,results[3],label='Recovered')
	plt.legend()
	plt.show()

def simulate_model_population(model):		
	T_max = 60
	dt = 1
	T = np.linspace(0,T_max,int(T_max/dt)+1)
	N = 1e8
	init_exposed = 10
	init_vals = (N-init_exposed),init_exposed,0,0
	alpha = 0.4
	beta = 1.0
	gamma1 = 0.0
	gamma2 = 0.01
	rho = 1.0
	params = alpha, beta*rho, gamma1, gamma2
	results = model(init_vals,params,T)
	plt.plot(T,results[0],label='Susceptible')
	plt.plot(T,results[1],label='Exposed')
	plt.plot(T,results[2],label='Infected')
	plt.plot(T,results[3],label='Recovered')
	plt.legend()
	plt.show()

if __name__=='__main__':
	# simulate_model(base_seir_model)
	simulate_model_population(corona_seir_model_population)