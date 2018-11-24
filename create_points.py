
import matplotlib.pyplot as plt
import numpy as np
import pickle

return 0

fig, ax = plt.subplots()

width = 100
height = 100
ax.plot([0, width, width, 0, 0], [0, 0, height, height, 0])

inp = "left"

plot = []
# plot = ["left"]
# plot = ["left", "right"]
# plot = ["left", "right","path"]

data = None
data = {
	"left": np.array([[0,0]]),
	"right": np.array([[0,0]]),
	"path": np.array([[0,0]])
}

# with open('track2.pkl', 'r') as handle:
# 	data = pickle.load(handle)

color = {
	"left": 'bs',
	"right": "rs",
	"path": "gs"
}

for pi in plot:
	p = data[pi]
	ax.plot(p[0:, 0], p[0:, 1], color[pi])



def onclick(event):
	x, y = event.xdata, event.ydata
	# data[inp].append((x, y, color))
	data[inp] = np.append(data[inp], [[x, y]], axis=0)

	ax.plot(x,y, color[inp])
	fig.canvas.draw()

fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()

print data
# Save
fig.savefig("track1.png")
with open('track2.pkl', 'wb') as handle:
	pickle.dump(data, handle)

