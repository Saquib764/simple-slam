
import matplotlib.pyplot as plt
import numpy as np

import math
from math import cos, sin, tan, pi, atan2, sqrt


N_PARTICLE = 2
LANDMARK_SIZE = 2 		#	[x,y]


Q = np.diag([3.0, np.deg2rad(10.0)])**2


def noise(cov):
	return cov*np.random.randn(len(cov))


class Particle:

	def __init__(self, N_LANDMARK):
		self.w = 1.0 / N_PARTICLE
		self.x = 0.0
		self.y = 0.0
		self.yaw = 0.0

		self.P = np.eye(3)
		# landmark x-y positions
		self.lm = np.matrix(np.zeros((N_LANDMARK, LANDMARK_SIZE)))
		# landmark position covariance
		self.lmP = np.matrix(np.zeros((N_LANDMARK * LANDMARK_SIZE, LANDMARK_SIZE)))


particles = [Particle(3) for i in range(N_PARTICLE)]

# print particles[0].lm

def init(start):
	global particles
	for particle in particles:
		particle.x = start[0]
		particle.y = start[1]
		particle.yaw = start[2]

def predict_particle(dt, motion_model, particles, u, R):
	for particle in particles:
		ur = u + noise(R)

		# state (xn, yn, thetan, w, vel), u -> (w, v)
		x = [particle.x, particle.y, particle.yaw, u[0], u[1]]

		xn = motion_model(dt, x, ur)
		particle.x = xn[0]
		particle.y = xn[1]
		particle.yaw = xn[2]

	return particles


def add_new_landmark(particle, z, Q):
	


def data_association(particle, z):
	index = []
	thres = 0.2
	for i in range(z):
		found = False
		for k in range(len(particle.lm[:, 0])):
			lm = particle.lm[k]
			if((lm[0] - z[i][0])**2 + (lm[1] - z[i][1])**2 < thres**2):
				found = True
				index.append((i, k))
				break;
		if !found:
			index.append(i, len(particle.lm[:, 0]))
			particle


def update_with_landmarks(particles, z):
	# z => [(d, angle), [x_actual,y_actual]]
	for i in range(len(z)):
		lm = z[i][0]
		print(i, lm)

	return particles

def fastSlam2(dt, u, landmarks, motion_model, R):
	global particles
	particles = predict_particle(dt, motion_model, particles, u, R)
	particles = update_with_landmarks(particles, landmarks)

	p = particles[0]
	return [ p.x,p.y, p.yaw, 0,0]