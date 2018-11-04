
import numpy as np
from math import cos, sin, tan, pi, sqrt, atan2, fabs
import matplotlib.pyplot as plt

import pickle
from agent import Agent

def rand(thres):
	return (1 + thres*2*(np.random.rand() - 0.5))


class World:
	def __init__(self, width, height, animation_frequency=200):
		self.width = width
		self.height = height
		self.animation_frequency = animation_frequency

		self.agents = []
		self.landmarks = []
		self.frequency = 100
		self.dt = 1./self.frequency

	def add_landmark(self, pos):
		# x, y, color
		self.landmarks.append(pos)

	def create_new_agent(self, color="red", state=(0,0,0), name="name"):
		a = Agent(self.dt, color=color, state=state, name=name)
		a.world = self
		self.agents.append(a)
		return a


	def plot(self):
		plt.clf()

		# plot border
		plt.plot([0, self.width, self.width, 0, 0], [0, 0, self.height, self.height, 0])

		#plot landmarks

		for l in self.landmarks:
			x, y, color = l
			plt.plot([x], [y], color)

		# Plot ground truth pos
		for a in self.agents:
			a.plot()


		plt.axis([0, self.width, 0, self.height])
		plt.grid(True)
		plt.pause(1./self.animation_frequency)

	def load(self, name):
		color = {
			"left": 'bs',
			"right": "rs",
			"path": "gs"
		}
		with open(name, 'r') as handle:
			data = pickle.load(handle)

			self.track = data

			for d in data["left"]:
				self.add_landmark((d[0], d[1], color["left"]))
			for d in data["right"]:
				self.add_landmark((d[0], d[1], color["right"]))
			for d in data["path"]:
				self.add_landmark((d[0], d[1], color["path"]))

			return data

	def create_control(self, motion_model, agent):
		# remove first
		path = self.track["path"][1:, 0:]

		# Cyclic
		path = np.append(path, [path[0]], axis=0)
		# path = np.append(path, path[1:5], axis=0)

		control = []
		x, y, th = path[0][0], path[0][1], 0
		for i in range(1, len(path)):
			# if i > 2:
			# 	break
			phi = 0
			c = 0
			while sqrt((path[i][1] - y)**2 + (path[i][0] - x)**2) > 5 or fabs(phi) < pi/2:
				# if i == 18 and c > 20:
				# 	break
				c+=1
				phi = atan2(path[i][1] - y, path[i][0] - x) - th

				if (phi < -pi ):
					phi = 2*pi + phi
				# phi = phi % (2*pi)

				phi = phi*rand(0.0)

				phi = min(pi/2, max(-pi/2, phi))
				# if i==18:
				# 	print phi, atan2(path[i][1] - y, path[i][0] - x), th, x, y, path[i]

				x, y, th = motion_model(agent.dt, x, y, th, agent.v, phi)
				control.append(phi)


		print len(control)
		return control
		

