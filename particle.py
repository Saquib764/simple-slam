
import numpy as np


import sys

import math
from math import cos, sin, tan, pi, atan2, sqrt


def gauss_likelihood(x, sigma):
	p = 1.0 / math.sqrt(2.0 * math.pi * sigma ** 2) * \
		math.exp(-x ** 2 / (2 * sigma ** 2))
	return p

def noise(cov):
	return cov*np.random.randn(len(cov))

N = 10
px = np.zeros((N, 5))
pw = np.zeros(N) + 1./N


def pi_2_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi

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



def pf(dt, u, landmarks, motion_model, cov):
	# [Lidar_sensor, real]
	global px, pw, N

	# print len(landmarks)

	for i in range(N):
		p = pw[i]
		x = px[i]

		ur = u + noise(cov)

		xn = motion_model(dt, x, ur)
		xn[2] = pi_2_pi(xn[2])

		for l in landmarks:
			reading = l[0]
			dx, dy = l[1][0]-xn[0], l[1][1]-xn[1]

			dangle = atan2(dy, dx) - xn[2] - reading[1]
			# print 50*dangle/180

			d = sqrt(dy**2 + dx**2) - reading[0]

			p = p*gauss_likelihood(d, 0.1)*gauss_likelihood(dangle, (50*pi/180))
			# p = p*gauss_likelihood(d, 0.1)

		px[i] = xn
		pw[i] = p



	if pw.sum() != 0.:
		pw = pw / pw.sum()
	else:
		# print "Zero"
		# sys.exit()
		pw = pw + 1./N

	xEst = np.matmul(pw, px)

	# print (pw*N).astype(int)

	resampling()
	return xEst

