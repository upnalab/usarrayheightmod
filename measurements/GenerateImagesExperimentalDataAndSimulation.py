import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ImageUtils import ImageUtils
from Waves import Waves

plt.close('all')

wavelength = 8.5 # mm
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

def targetPlot(target, name):
    plt.figure()
    plt.title('Target')
    plt.imshow(target, cmap = "Greys")
    plt.xticks(np.linspace(0, slicePx-1, 5), np.linspace(-sizeImageMm,sizeImageMm, 5, dtype = np.int16))
    plt.yticks(np.linspace(0, slicePx-1, 5), np.linspace(sizeImageMm,-sizeImageMm, 5, dtype = np.int16))
    plt.xlabel('x(mm)')
    plt.ylabel('y(mm)')
    
    plt.savefig(f'{name}.png', dpi = 1200)
    plt.show()

def simulation(sim, name):
    emittersPositions = np.empty(shape = (sim.shape[0], 3))
    emittersPositions[:,0] = sim.iloc[:,0]
    emittersPositions[:,1] = sim.iloc[:,1]
    emittersPositions[:,2] = sim.iloc[:,2]
    normals = Waves.constNormals(emittersPositions, [0,0,1])
    outputPositions = Waves.planeGridZ(0,0, distTarget, targetSize, targetSize, slicePx, slicePx)
    
    propagators = Waves.calcPropagatorsPistonsToPoints(emittersPositions, normals,outputPositions, k,  emitterApperture)
    
    field = np.abs(np.ones(emittersPositions.shape[0]) @ propagators).reshape([slicePx, slicePx])
    field = field / field.max()
    
    plt.figure()
    plt.title('Simulation')
    plt.imshow(field, cmap = plt.cm.gist_heat)
    plt.xticks(np.linspace(0, slicePx-1, 5), np.linspace(-sizeImageMm,sizeImageMm, 5, dtype = np.int16))
    plt.yticks(np.linspace(0, slicePx-1, 5), np.linspace(sizeImageMm,-sizeImageMm, 5, dtype = np.int16))
    plt.colorbar(label = "Normalized Amplitude")
    plt.xlabel('x(mm)')
    plt.ylabel('y(mm)')
    plt.savefig(f'{name}.png', dpi = 1200)
    plt.show()

def experimental(exper, name):
    plt.figure()
    if 'HeightMod' in name:
        plt.title('Experimental Data HeightMod')
    else:
        plt.title('Experimental Data PhaseMod')
    exper = exper / exper.max().max()
    plt.imshow(exper, origin = "lower", cmap = plt.cm.gist_heat)
    plt.xticks(np.linspace(0, exper.shape[1]-1, 5), np.linspace(-sizeImageMm, sizeImageMm, 5, dtype = np.int16))
    plt.yticks(np.linspace(0, exper.shape[0]-1, 5), np.linspace(-sizeImageMm, sizeImageMm, 5, dtype = np.int16))
    plt.colorbar(label = "Normalized Amplitude")
    plt.xlabel('x(mm)')
    plt.ylabel('y(mm)')
    plt.savefig(f'{name}.png', dpi = 1200)
    plt.show()

    
# Target
target_A = abs(1-ImageUtils.loadNorm("../patterns/A.png", slicePx))
targetPlot(target_A, '00_target_A')

# Simulation
sim_A = pd.read_csv('Simulations/heightMod_A.csv', header = None)
simulation(sim_A, '01_sim_A')

# Experimental data
experHeightMod_A = pd.read_csv('ExperimentalData_HeightMod/HeightMod_A.csv', header = None) # Muy Buena (rotar un poco y recortar)
experimental(experHeightMod_A, '02_experHeightMod_A')

# Experimental data
experPhaseMod_A = pd.read_csv('ExperimentalData_PhasedMod/PhaseMod_A.csv', header = None) # Muy Buena (rotar un poco y recortar)
experimental(experPhaseMod_A, '03_experPhaseMod_A')

# Target
target_12 = abs(1-ImageUtils.loadNorm("../patterns/12.png", slicePx))
targetPlot(target_12, '04_target_12')

# Simulation
sim_12 = pd.read_csv('Simulations/heightMod_12.csv', header = None)
simulation(sim_12, '05_sim_12')

# Experimental data
experHeightMod_12 = pd.read_csv('ExperimentalData_HeightMod/HeightMod_12.csv', header = None) # Buena
experimental(experHeightMod_12, '06_experHeightMod_12')

# Experimental data
experPhasedMod_12 = pd.read_csv('ExperimentalData_PhasedMod/PhaseMod_12.csv', header = None) # Buena
experimental(experPhasedMod_12, '07_experPhasedMod_12')

# Target
target_star = abs(1-ImageUtils.loadNorm("../patterns/star.png", slicePx))
targetPlot(target_star, '08_target_star')

# Simulation
sim_star = pd.read_csv('Simulations/heightMod_Star.csv', header = None)
simulation(sim_star, '09_sim_star')

# Experimental data
experHeightMod_star = pd.read_csv('ExperimentalData_HeightMod/HeightMod_Star.csv', header = None) # Buena (rotar un poco y recortar)
experimental(experHeightMod_star, '10_experHeightMod_star')
