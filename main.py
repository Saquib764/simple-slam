

import matplotlib.pyplot as plt
import numpy as np

import math
from math import cos, sin, tan, pi, atan2, sqrt

from world import World


import particle as P

def motion_model(dt, x, u):

	'''
	Returns a new state (xn, yn, thetan, w, vel), safety condition,
	and done condition, given an initial state (x, y, theta, vel)
	Numerical integration is done at a time step of dt [sec].
	u -> (w, v)


	Model
	x = A*x + B*u
	'''

	A = np.array([[1.0, 0, 0, 0, 0],
				   [0, 1.0, 0, 0, 0],
				   [0, 0, 1.0, 0, 0],
				   [0, 0, 0, 0, 0],
				   [0, 0, 0, 0, 0]])

	# state change
	theta = x[2]

	B = np.array([[0, dt*cos(theta)],
					[0, dt*sin(theta)],
					[dt, 0],
					[1, 0],
					[0, 1]])

	# new state (forward Euler integration)
	xn = np.matmul(A, x) + np.matmul(B, u)
	return xn


W = World(400, 400)
track = W.load("track1.pkl")

control_array = W.create_control(motion_model, W.dt)

# x -> (x, y, theta, w, v)
car = W.create_new_agent(color="green", state=W.start, name="Real")
car.C = np.array([[0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])
car.sensor_covariance = np.array([10*pi/180, 0.8])


ghost_particle = W.create_new_agent(color="red", state=W.start, name="Particle")

P.px[:] = W.start
Iterations = len(control_array)
for i in range(Iterations):

	# if i > 3:
	# 	break

	control = control_array[i]


	#	Find actual pos
	x = car.state
	xn = motion_model(W.dt, x, control)
	car.set(xn)


	# Read sensor
	y = car.get_observation()

	# Landmarks
	landmarks = car.get_landmark()


	# Particle filter localization
	u = y
	xn = P.pf(W.dt, u, landmarks, motion_model, car.sensor_covariance)
	ghost_particle.set(xn)


	if(i%W.frequency == 0):
		W.plot()




plt.show()

