import numpy as np
from math import cos, sin, tan, pi, sqrt, atan2, fabs
import matplotlib as mpl
import matplotlib.pyplot as plt

def dist(a, b):
	return sqrt( (a[0]-b[0])**2 + (a[1]-b[1])**2 )

def rand(thres):
	return thres*np.random.randn()

def noise(cov):
	return cov*np.random.randn(len(cov))

class Agent():
	"""docstring for Agent"""
	def __init__(self, dt, state, color="red", name="name"):
		# pos => x, y, theta
		self.name = name
		self.color = color
		self.state = state
		self.v = 5
		self.C = None
		self.dt = dt
		self.world = None

		self.range = 20

		self.odometry = 0

		# land mark => dist, theta
		self.observed_landmark = []

		self.history = np.array([state])

		self.sensor_covariance = None

		self.calculated = {}

	def plot(self):
		l = 5.
		x, y, th = self.state[0], self.state[1], self.state[2]
		# plt.plot([x], [y], "ro", markersize=8)

		t = mpl.markers.MarkerStyle(marker=">")
		t._transform = t.get_transform().rotate_deg(th*180/pi)

		plt.scatter((x), (y), marker=t, s= 100, color=self.color)
		# plt.plot(x, y, marker=(3, 1, th*180/pi - 90), markersize=20, linestyle='None')
		# plt.plot([x, x - l*cos(th)], [y, y - l*sin(th)], color="red", linewidth=2)

		# print self.history
		plt.plot(self.history[0:, 0], self.history[0:, 1], color=self.color, label=self.name)

		for l in self.observed_landmark:
			plt.plot([x, l[1][0]], [y, l[1][1]], color="black")

		
	def set(self, state):
		self.state = state
		self.history = np.append(self.history, [state], axis=0)
		# self.history.append([state[0], state[1]])

	def get_observation2(self):
		# simulate wheel encoder
		self.odometry = self.v + rand(0.8)
		self.u = self.yaw + rand(10*pi/180)
		return self.odometry, self.u

	def get_observation(self):
		# simulate wheel encoder and imu
		y = np.matmul(self.C, self.state) + noise(self.sensor_covariance)
		return y


	def get_landmark(self):
		# simulates LidaR
		self.observed_landmark = []
		for l in self.world.landmarks:
			d = dist(self.state, l)
			if d <= 20:
				angle = atan2(l[1] - self.state[1], l[0] - self.state[0]) - self.state[2]
				if(fabs(angle) > 150*pi/360):
					continue

				# induce noise
				d = d + rand(0.2)
				angle = angle + rand(10*pi/180)

				# errored, actaul x-y
				self.observed_landmark.append([(d, angle), l])

		# print self.observed_landmark

		return self.observed_landmark


