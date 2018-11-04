
import matplotlib.pyplot as plt
import numpy as np

import math
from math import cos, sin, tan, pi, atan2, sqrt

from world import World

"""
TODO: Path creation module. input red come, blue cone, waypoint by clicking and pickle it

"""

def rand(thres):
	return (0 + thres*2*(np.random.rand() - 0.5))


def motion_model_step(dt, x, y, theta, v, w):

	'''
	Returns a new state (xn, yn, thetan), safety condition,
	and done condition, given an initial state (x, y, theta)
	and control w is yaw rate.
	Numerical integration is done at a time step of dt [sec].
	'''

	# state rate
	dx     = v*cos(theta)
	dy     = v*sin(theta)
	dtheta = w

	# new state (forward Euler integration)
	xn     = x     + dt*dx
	yn     = y     + dt*dy
	thetan = theta + dt*dtheta

	return xn, yn, thetan


def gauss_likelihood(x, sigma):
	p = 1.0 / math.sqrt(2.0 * math.pi * sigma ** 2) * \
		math.exp(-x ** 2 / (2 * sigma ** 2))
	return p

N = 100
px = np.zeros((N, 3))
pw = np.zeros(N) + 1./N


def resampling():
	"""
	low variance re-sampling
	"""

	global px, pw
	NP = N
	NTh = 0.5*N

	Neff = 1.0 / (np.matmul(pw, pw))  # Effective particle number
	if Neff < NTh:
		wcum = np.cumsum(pw)
		base = np.cumsum(pw * 0.0 + 1 / NP) - 1 / NP
		resampleid = base + np.random.rand(base.shape[0])
		# resampleid = np.cumsum(resampleid)
		# print wcum, resampleid, base, (pw*N).astype(int)

		inds = []
		
		for ip in range(NP):
			ind = 0
			while resampleid[ip] > wcum[ind]:
				ind += 1
			inds.append(ind)

		px = px[inds, 0:]
		pw = np.zeros(NP) + 1.0 / NP  # init weight

		# print inds

	# print px, pw



def pf(dt, v, w, landmarks):
	global px, pw

	for i in range(N):
		p = pw[i]
		x = px[i]

		uv = v + rand(0.8)
		uw = w + rand(10*pi/180)

		nx, ny, ntheta = motion_model_step(dt, x[0], x[1], x[2], uv, uw)

		for l in landmarks:
			reading = l[0]
			dx, dy = l[1][0]-nx, l[1][1]-ny
			dangle = atan2(dy, dx) - ntheta - reading[1]
			d = sqrt(dy**2 + dx**2) - reading[0]

			p = p*gauss_likelihood(d, 0.1)*gauss_likelihood(dangle, (50*pi/180))

		px[i] = [nx, ny, ntheta]
		pw[i] = p


	
	pw = pw / pw.sum()

	xEst = np.matmul(pw, px)

	# print (pw*N).astype(int)

	resampling()

	return xEst[0], xEst[1], xEst[2]




W = World(400, 400)
track = W.load("track1.pkl")


start = (track["path"][1][0], track["path"][1][1], 0)
car = W.create_new_agent(color="red", state=start, name="Real")

control_array = W.create_control(motion_model_step, car)

ghost_dead_reckoning = W.create_new_agent(color="green", state=start, name="dead reckoning")


# start = (track["path"][1][0] + 10 , track["path"][1][1] + 10, 0)

ghost_particle = W.create_new_agent(color="black", state=start, name="particle")

px[0:] = [track["path"][1][0], track["path"][1][1], 0]

Iterations = len(control_array)
for i in range(Iterations):

	# if i > 3:
	# 	break

	car.yaw = control_array[i]


	#	Find actual pos
	x, y, th = car.ground_state
	xn, yn, thn = motion_model_step(W.dt, x, y, th, car.v, car.yaw)
	car.set((xn, yn, thn))


	# Do odometry and sensor reading
	v, yaw = car.get_observation()

	# print v, yaw, car.v, car.yaw
	landmark = car.get_landmark()


	# Find xEst particle filter
	nx, ny, nth = pf(W.dt, v, yaw, landmark)
	ghost_particle.set((nx, ny, nth))


	dif = np.array([nx-x, ny-y, nth-th])
	print np.linalg.norm(dif)



	# update dead reckoning
	ghost_dead_reckoning.v = v
	ghost_dead_reckoning.yaw = yaw
	x, y, th = ghost_dead_reckoning.ground_state
	xn, yn, thn = motion_model_step(W.dt, x, y, th, ghost_dead_reckoning.v, ghost_dead_reckoning.yaw)
	ghost_dead_reckoning.set((xn, yn, thn))


	if(i%W.frequency == 0):
		W.plot()


plt.show()