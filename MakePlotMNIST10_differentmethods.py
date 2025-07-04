import numpy as np
from Waves import Waves
from ImageUtils import ImageUtils
from ArrayAmpSliceHeight import ArrayAmpSliceHeight
import os
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm 

plt.close('all')

arraySize = 0.16
targetSize = 0.16
emittersPerSide = 16
nEmitters = emittersPerSide * emittersPerSide
emitterApperture = 0.009
slicePx = 192 # 256
distTarget = 0.16
c = 340
fr = 40000
wavelength = c / fr 
k = 2 * np.pi / wavelength

dir_ = 'patterns/mnist10/'

dic = {
       
       'Height Mod'             : ( True , False, False ), 
       'Phase Mod'              : ( False, True , False ), 
       'Height+Amp Mod'         : ( True , False, True  ), 
       'Phase+Amp Mod'          : ( False, True , True  ), 
       'Height+Phase Mod'       : ( True , True , False ), 
       'Height+Phase+Amp Mod'   : ( True , True , True  ), 
       
       }

dic_values = { 'Height Mod' : [], 'Phase Mod' : [], 'Height+Amp Mod' : [], 'Phase+Amp Mod' : [], 'Height+Phase Mod' : [], 'Height+Phase+Amp Mod' : [] }

for key in tqdm(list(dic.keys())[0:2]):
    
    optimizeHeight, optimizePhase, optimizeAmp = dic[key]
    
    for name in os.listdir(dir_)[:1]:
        
        import time
        a = time.time()
        
        #targets
        target = ImageUtils.loadNorm(dir_ + name, slicePx)
        
        lX = np.linspace(0 - arraySize / 2, 0 + arraySize / 2, emittersPerSide)
        lY = np.linspace(0 - arraySize / 2, 0 + arraySize / 2, emittersPerSide)
        coordsXY = np.meshgrid(lX, lY)
        coordsXY = np.vstack([np.ravel(coord) for coord in coordsXY]).T
        
        outputPositions = Waves.planeGridZ(0,0, distTarget, targetSize, targetSize, slicePx, slicePx)
        
        opti = ArrayAmpSliceHeight()
        opti.iters = 1_000 # 240
        opti.showLossEvery = 100 # opti.iters + 1
        opti.learningRate = 0.1
        
        loss, heights, phases, emitterPositions, ampField, amps = opti.optimizeAmpSlice(target, distTarget, outputPositions, coordsXY, nEmitters, 
                                                                                        optimizeHeight = optimizeHeight, optimizePhase = optimizePhase, optimizeAmp = optimizeAmp)
        
        plt.figure()
        plt.imshow(ampField, cmap = plt.cm.gist_heat)
        plt.colorbar()
        # plt.savefig(f'Results_MNIST10/{key}/{name[:-5]}.svg', format='svg')
        plt.show()
        # plt.close('all')
        
        df = pd.DataFrame(ampField)
        # df.to_csv(f'Results_MNIST10/{key}/data/{name[:-5]}.csv', header = False)
    
        dic_values[key].append(loss)
        
        b = time.time()
        print(f'He  tardado {b-a} segundos en ejecutar')
        
        
        

plt.figure()
values = [v for v in dic_values.values()]
plt.boxplot(values, patch_artist=True)
plt.xticks(range(1,len(values)+1), list(dic.keys()), rotation = 20)
plt.savefig('Results_MNIST10/BoxPlot.svg', format='svg')
plt.show()