import numpy as np
from math import cos, sin, tan, pi, sqrt, atan2
import matplotlib as mpl
import matplotlib.pyplot as plt

def dist(a, b):
	return sqrt( (a[0]-b[0])**2 + (a[1]-b[1])**2 )

def rand(thres):
	return (0 + thres*2*(np.random.rand() - 0.5))

class Agent():
	"""docstring for Agent"""
	def __init__(self, dt, color="red", state = (0,0,0), name="name"):
		# pos => x, y, theta
		self.name = name
		self.color = color
		self.ground_state = state
		self.v = 5
		self.dt = dt
		self.world = None

		self.range = 20

		self.odometry = 0

		# land mark => dist, theta
		self.observed_landmark = []

		self.history = np.array([[state[0], state[1]]])

	def plot(self):
		l = 5.
		x, y, th = self.ground_state
		# plt.plot([x], [y], "ro", markersize=8)

		t = mpl.markers.MarkerStyle(marker=">")
		t._transform = t.get_transform().rotate_deg(th*180/pi)

		plt.scatter((x), (y), marker=t, s= 100, color=self.color)
		# plt.plot(x, y, marker=(3, 1, th*180/pi - 90), markersize=20, linestyle='None')
		# plt.plot([x, x - l*cos(th)], [y, y - l*sin(th)], color="red", linewidth=2)

		# print self.history
		plt.plot(self.history[0:, 0], self.history[0:, 1], color=self.color, label=self.name)
		
	def set(self, state):
		self.ground_state = state
		self.history = np.append(self.history, [[state[0], state[1]]], axis=0)
		# self.history.append([state[0], state[1]])

	def get_observation(self):
		# simulate wheel encoder
		self.odometry = self.v + rand(0.8)
		self.u = self.yaw + rand(10*pi/180)
		return self.odometry, self.u


	def get_landmark(self):
		# simulates LidaR
		self.observed_landmark = []
		for l in self.world.landmarks:
			d = dist(self.ground_state, l)
			if d <= 30:
				angle = atan2(l[1] - self.ground_state[1], l[0] - self.ground_state[0]) - self.ground_state[2]

				# induce noise
				d = d + rand(0.2)
				angle = angle + rand(10*pi/180)

				# errored, actaul x-y
				self.observed_landmark.append([(d, angle), l])

		return self.observed_landmark


