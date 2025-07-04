import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
sys.path.append("../..")
from Waves import Waves

plt.close('all')

sizeImageMm = 80 
slicePx = 256
# Simulations
distTarget = 0.16
targetSize = 0.16
emitterApperture = 0.009
c = 340
fr = 40000
wavelength = c / fr 
k = 2 * np.pi / wavelength


for file in os.listdir():
    
    if not '.csv' in file:
        continue
    
    data = pd.read_csv(file, header = None)
    data.columns = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'power', 'frequency', 'apperture', 'type', 'sx', 'sy', 'sz', 'phase']
    
    emittersPositions = np.empty(shape = (data.shape[0], 3))
    emittersPositions[:,0] = data['x']
    emittersPositions[:,1] = data['y']
    emittersPositions[:,2] = data['z']
    
    normals = np.empty(shape = (data.shape[0], 3))
    normals[:,0] = data['nx']
    normals[:,1] = data['ny']
    normals[:,2] = data['nz']
    
    signal = np.exp(1j * data['phase'])
    
    targetPositions = Waves.planeGridZ(0,0, distTarget, targetSize, targetSize, slicePx, slicePx)
    propagators = Waves.calcPropagatorsPistonsToPoints(emittersPositions, normals,targetPositions, k,  emitterApperture)
    
    field = signal @ propagators
    amplitude = np.abs(field)
    amplitude_norm = amplitude / amplitude.max()
    amplitude_norm_reshape = amplitude_norm.reshape([slicePx, slicePx])
    
    ax = plt.figure().add_subplot(projection='3d')
    ax.set_title(file)
    scatter_em = ax.scatter(emittersPositions[:,0], emittersPositions[:,1], emittersPositions[:,2], c = np.angle(signal), cmap = plt.cm.hsv)
    scatter_amp = ax.scatter(targetPositions[:,0], targetPositions[:,1], targetPositions[:,2], c = amplitude_norm_reshape, cmap = plt.cm.gist_heat)
    
    ax.set_xticks(np.linspace(emittersPositions[:,0].min(), emittersPositions[:,0].max(), 5))
    ax.set_yticks(np.linspace(emittersPositions[:,1].min(), emittersPositions[:,1].max(), 5))
    ax.set_zticks(np.linspace(emittersPositions[:,2].min(), targetPositions[:,2].max(), 5))
    
    phase_cbar = plt.colorbar(scatter_em, label = "Phase", ax = ax)
    ticks_phase = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
    tick_labels = [r"$-\pi$", r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$", r"$\pi$"]
    phase_cbar.set_ticks(ticks_phase)
    phase_cbar.set_ticklabels(tick_labels)
    amp_cbar = plt.colorbar(scatter_amp, label = "Normalized Amplitude", ax = ax)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    plt.show()
