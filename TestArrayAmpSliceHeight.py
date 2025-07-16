import matplotlib.pyplot as plt
from ImageUtils import ImageUtils
from Waves import Waves
from ArrayAmpSliceHeight import ArrayAmpSlice
import numpy as np
import pandas as pd

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

def execute_model(cPath, path):
    target = ImageUtils.loadNorm(cPath + path + ".png", slicePx)
    
    opti = ArrayAmpSlice()
    opti.iters = 1500
    opti.showLossEvery = 100 #opti.iters + 1
    loss, heights, phases, emitterPositions, ampField, amps = opti.optimizeAmpSlice(target, distTarget, outputPositions, coordsXY, nEmitters)
    
    # Plot the example
    plt.figure()
    plt.tilte('Simulation')
    plt.imshow(ampField, cmap = plt.cm.gist_heat, extent=[outputPositions[:,0].min(),outputPositions[:,0].max(),outputPositions[:,1].min(),outputPositions[:,1].max()])
    plt.colorbar(label = "Normalized Amplitude")
    plt.xlabel('x(mm)')
    plt.ylabel('y(mm)')
    plt.savefig(f'sim_{path}.png', dpi = 1200)
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
    
    # Save to csv
    # File format: x,y,z,nx,ny,nz,power,frequency,apperture,Type(0=circle,1=square),sx,sy,sz,phase
    df = pd.DataFrame(data = {
        'x'         : emitterPositions[:, 0],
        'y'         : emitterPositions[:, 1],
        'z'         : emitterPositions[:, 2],
        'nx'        : np.zeros(emitterPositions.shape[0]),
        'ny'        : np.zeros(emitterPositions.shape[0]),
        'nz'        : np.ones(emitterPositions.shape[0]),
        'power'     : np.ones(emitterPositions.shape[0]) * 2.4,
        'frequency' : np.ones(emitterPositions.shape[0]) * 40_000,
        'apperture' : np.ones(emitterPositions.shape[0]) * 0.009,
        'type'      : np.zeros(emitterPositions.shape[0]),
        'sx'        : np.ones(emitterPositions.shape[0]) * 0.01,
        'sy'        : np.ones(emitterPositions.shape[0]) * 0.01,
        'sz'        : np.ones(emitterPositions.shape[0]) * 0.003,
        'phase'     : np.ones(emitterPositions.shape[0]) * phases.numpy().flatten() * np.pi
    })
    
    df.to_csv(f'measurements/Simulations/heightMod_{path}.csv', sep = ',', header = False, index = False)
    
# Target A
cPath = "patterns/"
path = "A" 
execute_model(cPath, path)


# Target 12
cPath = "patterns/"
path = "12" 
execute_model(cPath, path)

# Target Star
cPath = "patterns/"
path = "star" 
execute_model(cPath, path)