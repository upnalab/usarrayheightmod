import matplotlib.pyplot as plt
from ImageUtils import ImageUtils
from Waves import Waves
from ArrayAmpSliceHeight import ArrayAmpSlice
import numpy as np

arraySize = 0.16
targetSize = 0.16
emittersPerSide = 16
nEmitters = emittersPerSide * emittersPerSide
emitterApperture = 0.009
slicePx = 128
distTarget = 0.16
c = 340
fr = 40000
wavelength = c / fr 
k = 2 * np.pi / wavelength

lX = np.linspace(0 - arraySize / 2, 0 + arraySize / 2, emittersPerSide)
lY = np.linspace(0 - arraySize / 2, 0 + arraySize / 2, emittersPerSide)
coordsXY = np.meshgrid(lX, lY)
coordsXY = np.vstack([np.ravel(coord) for coord in coordsXY]).T

outputPositions = Waves.planeGridZ(0,0, distTarget, targetSize, targetSize, slicePx, slicePx)

# Target A
cPath = "patterns/"
path = "A" 
target = ImageUtils.loadNorm(cPath + path + ".png", slicePx)

opti = ArrayAmpSlice()
opti.iters = 1500
opti.showLossEvery = 100 #opti.iters + 1
loss, heights, phases, emitterPositions, ampField, amps = opti.optimizeAmpSlice(target, distTarget, outputPositions, coordsXY, nEmitters)

# Plot the example
plt.figure()
plt.imshow(ampField, cmap = plt.cm.gist_heat, extent=[outputPositions[:,0].min(),outputPositions[:,0].max(),outputPositions[:,1].min(),outputPositions[:,1].max()])
plt.colorbar(label = "Normalized Amplitude")
plt.xlabel('x(mm)')
plt.ylabel('y(mm)')
plt.show()

# Plot the example in 3D
ax = plt.figure().add_subplot(projection='3d')
cs2 = ax.scatter(emitterPositions[:,0], emitterPositions[:,1], emitterPositions[:,2], c = phases, alpha = np.round(amps,5), cmap = plt.cm.hsv, vmin = -np.pi, vmax = np.pi)
cs  = ax.scatter(outputPositions[:,0],  outputPositions[:,1],  outputPositions[:,2], c = ampField, cmap = plt.cm.gist_heat)
cbar = plt.colorbar(cs2, ax = ax, label='Phase (rad)')
ticks = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
tick_labels = [r"$-\pi$", r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$", r"$\pi$"]

cbar.set_ticks(ticks)
cbar.set_ticklabels(tick_labels)

ax.set_zticks(np.arange(0, distTarget+1e-2, 2e-2))
ax.set_xticks(np.arange(outputPositions[:,0].min(),outputPositions[:,0].max()+1e-2, 2e-2))
ax.set_yticks(np.arange(outputPositions[:,1].min(),outputPositions[:,1].max()+1e-2, 2e-2))
plt.show()

# Target 12
cPath = "patterns/"
path = "12" 
target = ImageUtils.loadNorm(cPath + path + ".png", slicePx)

opti = ArrayAmpSlice()
opti.iters = 1500
opti.showLossEvery = 100 #opti.iters + 1
loss, heights, phases, emitterPositions, ampField, amps = opti.optimizeAmpSlice(target, distTarget, outputPositions, coordsXY, nEmitters)

# Plot the example
plt.figure()
plt.imshow(ampField, cmap = plt.cm.gist_heat, extent=[outputPositions[:,0].min(),outputPositions[:,0].max(),outputPositions[:,1].min(),outputPositions[:,1].max()])
plt.colorbar(label = "Normalized Amplitude")
plt.xlabel('x(mm)')
plt.ylabel('y(mm)')
plt.show()

# Plot the example in 3D
ax = plt.figure().add_subplot(projection='3d')
cs2 = ax.scatter(emitterPositions[:,0], emitterPositions[:,1], emitterPositions[:,2], c = phases, alpha = np.round(amps,5), cmap = plt.cm.hsv, vmin = -np.pi, vmax = np.pi)
cs  = ax.scatter(outputPositions[:,0],  outputPositions[:,1],  outputPositions[:,2], c = ampField, cmap = plt.cm.gist_heat)
cbar = plt.colorbar(cs2, ax = ax, label='Phase (rad)')
ticks = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
tick_labels = [r"$-\pi$", r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$", r"$\pi$"]

cbar.set_ticks(ticks)
cbar.set_ticklabels(tick_labels)

ax.set_zticks(np.arange(0, distTarget+1e-2, 2e-2))
ax.set_xticks(np.arange(outputPositions[:,0].min(),outputPositions[:,0].max()+1e-2, 2e-2))
ax.set_yticks(np.arange(outputPositions[:,1].min(),outputPositions[:,1].max()+1e-2, 2e-2))
plt.show()

# Target Star
cPath = "patterns/"
path = "star" 
target = ImageUtils.loadNorm(cPath + path + ".png", slicePx)

opti = ArrayAmpSlice()
opti.iters = 1500
opti.showLossEvery = 100 #opti.iters + 1
loss, heights, phases, emitterPositions, ampField, amps = opti.optimizeAmpSlice(target, distTarget, outputPositions, coordsXY, nEmitters)

# Plot the example
plt.figure()
plt.imshow(ampField, cmap = plt.cm.gist_heat, extent=[outputPositions[:,0].min(),outputPositions[:,0].max(),outputPositions[:,1].min(),outputPositions[:,1].max()])
plt.colorbar(label = "Normalized Amplitude")
plt.xlabel('x(mm)')
plt.ylabel('y(mm)')
plt.show()

# Plot the example in 3D
ax = plt.figure().add_subplot(projection='3d')
cs2 = ax.scatter(emitterPositions[:,0], emitterPositions[:,1], emitterPositions[:,2], c = phases, alpha = np.round(amps,5), cmap = plt.cm.hsv, vmin = -np.pi, vmax = np.pi)
cs  = ax.scatter(outputPositions[:,0],  outputPositions[:,1],  outputPositions[:,2], c = ampField, cmap = plt.cm.gist_heat)
cbar = plt.colorbar(cs2, ax = ax, label='Phase (rad)')
ticks = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
tick_labels = [r"$-\pi$", r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$", r"$\pi$"]

cbar.set_ticks(ticks)
cbar.set_ticklabels(tick_labels)

ax.set_zticks(np.arange(0, distTarget+1e-2, 2e-2))
ax.set_xticks(np.arange(outputPositions[:,0].min(),outputPositions[:,0].max()+1e-2, 2e-2))
ax.set_yticks(np.arange(outputPositions[:,1].min(),outputPositions[:,1].max()+1e-2, 2e-2))
plt.show()