import numpy as np
import matplotlib.pyplot as plt

def seir_model(init_vals,params,T):
	S0,E0,I0,R0 = init_vals
	S, E, I , R = [S0], [E0], [I0], [R0]
	alpha, beta, gamma = params
	dt = T[1]-T[0]
	for t in T[1:]:
		S1 = S[-1] - (beta*S[-1]*I[-1])*dt
		E1 = E[-1] + (beta*S[-1]*I[-1] - alpha*E[-1])*dt
		I1 = I[-1] + (alpha*E[-1] - gamma*I[-1])*dt
		R1 = R[-1] + (gamma*I[-1])*dt
		S.append(S1)
		E.append(E1)
		I.append(I1)
		R.append(R1)
	return np.stack([S,E,I,R])

if __name__=='__main__':
	T_max = 100
	dt = 1
	T = np.linspace(0,T_max,int(T_max/dt)+1)
	N = 10000
	init_exposed = 5
	init_vals = (N-init_exposed)/N,init_exposed/N,0,0
	alpha = 0.2
	beta = 1.75
	gamma = 0.5
	params = alpha, beta, gamma
	results = seir_model(init_vals,params,T)
	# plt.plot(T,results[0],label='Susceptible')
	plt.plot(T,results[1]*N,label='Exposed')
	plt.plot(T,results[2]*N,label='Infected')
	# plt.plot(T,results[3],label='Recovered')
	plt.legend()
	plt.show()