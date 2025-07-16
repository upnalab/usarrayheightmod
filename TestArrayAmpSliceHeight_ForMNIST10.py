import numpy as np
from Waves import Waves
from ImageUtils import ImageUtils
from ArrayAmpSliceHeight import ArrayAmpSlice
import open3d as o3d
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import pandas as pd

arraySize = 0.16
targetSize = 0.16
emittersPerSide = 16
nEmitters = emittersPerSide * emittersPerSide
emitterApperture = 0.009
slicePx = 256
distTarget = 0.16
c = 340
fr = 40000
wavelength = c / fr 
k = 2 * np.pi / wavelength

dic = {
       
       'Height Mod'             : ( True , False, False ), 
       'Phase Mod'              : ( False, True , False ), 
       'Height+Amp Mod'         : ( True , False, True  ), 
       'Phase+Amp Mod'          : ( False, True , True  ), 
       'Height+Phase Mod'       : ( True , True , False ), 
       'Height+Phase+Amp Mod'   : ( True , True , True  ), 
       
       }

dic_values = { 'Height Mod' : [], 'Phase Mod' : [], 'Height+Amp Mod' : [], 'Phase+Amp Mod' : [], 'Height+Phase Mod' : [], 'Height+Phase+Amp Mod' : [] }

dir_ = 'mnist10/'

fieldsList              = []
emitterPositionList     = []
heightList              = []
phasesList              = []
ampsList                = []
max_valueList           = []

for key in tqdm(list(dic.keys())):
    

    optimizeHeight, optimizePhase, optimizeAmp = dic[key]

    for name in os.listdir(dir_)[:1]:
        
        #targets
        target = ImageUtils.loadNorm(dir_ + name, slicePx)
        
        lX = np.linspace(0 - arraySize / 2, 0 + arraySize / 2, emittersPerSide)
        lY = np.linspace(0 - arraySize / 2, 0 + arraySize / 2, emittersPerSide)
        coordsXY = np.meshgrid(lX, lY)
        coordsXY = np.vstack([np.ravel(coord) for coord in coordsXY]).T
        
        outputPositions = Waves.planeGridZ(0,0, distTarget, targetSize, targetSize, slicePx, slicePx)
        
        opti = ArrayAmpSlice()
        opti.iters = 1_000
        opti.showLossEvery = 20 # opti.iters + 1
        
        loss, heights, phases, emitterPositions, ampField, amps = opti.optimizeAmpSlice(target, distTarget, outputPositions, coordsXY, nEmitters, 
                                                                                        optimizeHeight = optimizeHeight, optimizePhase = optimizePhase, optimizeAmp = optimizeAmp)
        
        fieldsList.append(ampField)
        emitterPositionList.append(emitterPositions)
        heightList.append(heights)
        phasesList.append(phases)
        ampsList.append(amps)
        max_valueList.append(np.array(ampField).max())
        
        plt.figure()
        plt.imshow(ampField, cmap = plt.cm.gist_heat)
        plt.colorbar()
        plt.savefig(f'Results_MNIST10/{key}/{name[:-5]}.svg', format='svg')
        plt.show()
        plt.close('all')
    
        df = pd.DataFrame(ampField)
        df.to_csv(f'Results_MNIST10/{key}/data/Field/{name[:-5]}.csv', header = False)
        df = pd.DataFrame(phases)
        df.to_csv(f'Results_MNIST10/{key}/data/Phases/{name[:-5]}.csv', header = False)
        df = pd.DataFrame(phases)
        df.to_csv(f'Results_MNIST10/{key}/data/Heights/{name[:-5]}.csv', header = False)
        df = pd.DataFrame(phases)
        df.to_csv(f'Results_MNIST10/{key}/data/Amps/{name[:-5]}.csv', header = False)
        
        dic_values[key].append(loss)
        
        
        ax = plt.figure().add_subplot(projection='3d')
        cs2 = ax.scatter(emitterPositions[:,0], emitterPositions[:,1], emitterPositions[:,2], c = phases, alpha = np.round(amps,5), cmap = plt.cm.hsv, vmin = -np.pi, vmax = np.pi)
        cs  = ax.scatter(outputPositions[:,0],  outputPositions[:,1],  outputPositions[:,2], c = ampField, cmap = plt.cm.gist_heat)
        cbar = plt.colorbar(cs2, ax = ax, label='Phase (rad)')
        ticks = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
        tick_labels = [r"$-\pi$", r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$", r"$\pi$"]
  
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(tick_labels)
        
        ax.set_zticks(np.arange(0,17e-2,2e-2))
        ax.set_xticks(np.arange(-8e-2,9e-2, 2e-2))
        ax.set_yticks(np.arange(-8e-2,9e-2, 2e-2))
        plt.savefig(f'Results_MNIST10/{key}/{name[:-5]}_3D.png', format='png')
        plt.show()
    
        plt.close('all')
        
        
for i, key in enumerate(list(dic.keys())):
    
    for name in os.listdir(dir_)[:1]:
        ampField = fieldsList[i]
        emitterPositions = emitterPositionList[i]
        heights = heightList[i]
        phases = phasesList[i]
        amps = ampsList[i]
        
        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter(emitterPositions[:,0], emitterPositions[:,1], emitterPositions[:,2], c = phases, alpha = np.round(amps,5), cmap = plt.cm.hsv, vmin = -np.pi, vmax = np.pi)
        cs = ax.scatter(outputPositions[:,0],  outputPositions[:,1],  outputPositions[:,2], c = ampField / np.max(max_valueList), cmap = plt.cm.gist_heat)
        plt.colorbar(cs)
        ax.set_zticks(np.arange(0,17e-2,2e-2))
        ax.set_xticks(np.arange(-8e-2,9e-2, 2e-2))
        ax.set_yticks(np.arange(-8e-2,9e-2, 2e-2))
        plt.savefig(f'Results_MNIST10/{key}/{name[:-5]}_3D_Norm.png', format='png')
        plt.show()
        
        plt.close('all')
    