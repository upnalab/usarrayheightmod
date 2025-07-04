import numpy as np
from Waves import Waves
from ImageUtils import ImageUtils
# from ArrayAmpSliceHeight import ArrayAmpSliceHeight
from ArrayAmpSliceHeight_Antiguo import ArrayAmpSlice
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

# #targets
# target = ImageUtils.loadNorm("mnist10/Label_0.jpeg", slicePx) # ImageUtils.loadNorm("thickA.png", slicePx)

# lX = np.linspace(0 - arraySize / 2, 0 + arraySize / 2, emittersPerSide)
# lY = np.linspace(0 - arraySize / 2, 0 + arraySize / 2, emittersPerSide)
# coordsXY = np.meshgrid(lX, lY)
# coordsXY = np.vstack([np.ravel(coord) for coord in coordsXY]).T

# outputPositions = Waves.planeGridZ(0,0, distTarget, targetSize, targetSize, slicePx, slicePx)

# lossesHeightMod = []
# lossesPhaseMod = []
# lossesPhaseAmpMod = []
# # ArrayAmpSlice 16x16

# # opti = ArrayAmpSliceHeight()
# opti = ArrayAmpSlice()
# loss, heights, phases, emitterPositions, ampField, cosAmps = opti.optimizeAmpSlice(target, distTarget, outputPositions, coordsXY, nEmitters)


# list_to_show = []
# for em in range(len(emitterPositions)):
#     emitterStl = o3d.io.read_triangle_mesh("Transductor_10mm.stl")
#     emitterStl = emitterStl.compute_vertex_normals()
#     emitterStl_vertices = np.asarray(emitterStl.vertices) * 0.001


#     vertices_circleEm = emitterStl_vertices[emitterStl_vertices[:,2] == emitterStl_vertices[:,2].max()]
#     center_em = vertices_circleEm.mean(axis = 0)
    
#     emitterStl_vertices = emitterStl_vertices + ( emitterPositions[em] - center_em)
    
#     emitterStl.vertices = o3d.utility.Vector3dVector(emitterStl_vertices)
    
#     list_to_show.append(emitterStl)
        

# colors = np.zeros(outputPositions.shape)
# colorsEmitters=np.zeros([emitterPositions.shape[0],3])
# colorsEmitters[:,1]=1
# ampField = np.array(ampField).flatten()
# colors[:,0] = ampField/ampField.max()

# emitters_cloud = o3d.geometry.PointCloud()
# emitters_cloud.points = o3d.utility.Vector3dVector(emitterPositions)
# emitters_cloud.colors = o3d.utility.Vector3dVector(colorsEmitters)
# list_to_show.append(emitters_cloud)

# outputPositions_cloud = o3d.geometry.PointCloud()
# outputPositions_cloud.points = o3d.utility.Vector3dVector(outputPositions)
# outputPositions_cloud.colors = o3d.utility.Vector3dVector(colors)
# list_to_show.append(outputPositions_cloud)

# o3d.visualization.draw_geometries(list_to_show)


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
        
        import time
        a = time.time()
        
        #targets
        target = ImageUtils.loadNorm(dir_ + name, slicePx)
        
        lX = np.linspace(0 - arraySize / 2, 0 + arraySize / 2, emittersPerSide)
        lY = np.linspace(0 - arraySize / 2, 0 + arraySize / 2, emittersPerSide)
        coordsXY = np.meshgrid(lX, lY)
        coordsXY = np.vstack([np.ravel(coord) for coord in coordsXY]).T
        
        outputPositions = Waves.planeGridZ(0,0, distTarget, targetSize, targetSize, slicePx, slicePx)
        
        opti = ArrayAmpSlice()
        opti.iters = 1_000
        opti.showLossEvery = 100 # opti.iters + 1
        
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
        
        b = time.time()
        print(f'He  tardado {b-a} segundos en ejecutar')
        
        
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
    
# plt.figure()
# values = [v for v in dic_values.values()]
# plt.boxplot(values, patch_artist=True)
# plt.xticks(range(1,len(values)+1), list(dic.keys()), rotation = 20)
# plt.savefig('Results_MNIST10/BoxPlot.svg', format='svg')
# plt.show()


# dic_values =
# {'Height Mod': [<tf.Tensor: shape=(), dtype=float32, numpy=0.00929157>,
#   <tf.Tensor: shape=(), dtype=float32, numpy=0.007287032>,
#   <tf.Tensor: shape=(), dtype=float32, numpy=0.0088616945>,
#   <tf.Tensor: shape=(), dtype=float32, numpy=0.010495214>,
#   <tf.Tensor: shape=(), dtype=float32, numpy=0.0069400663>,
#   <tf.Tensor: shape=(), dtype=float32, numpy=0.01009256>,
#   <tf.Tensor: shape=(), dtype=float32, numpy=0.0073677977>,
#   <tf.Tensor: shape=(), dtype=float32, numpy=0.008680126>,
#   <tf.Tensor: shape=(), dtype=float32, numpy=0.012613188>,
#   <tf.Tensor: shape=(), dtype=float32, numpy=0.00875471>],
#  'Phase Mod': [<tf.Tensor: shape=(), dtype=float32, numpy=0.0073242886>,
#   <tf.Tensor: shape=(), dtype=float32, numpy=0.006439584>,
#   <tf.Tensor: shape=(), dtype=float32, numpy=0.008098103>,
#   <tf.Tensor: shape=(), dtype=float32, numpy=0.009347353>,
#   <tf.Tensor: shape=(), dtype=float32, numpy=0.0079409685>,
#   <tf.Tensor: shape=(), dtype=float32, numpy=0.009769997>,
#   <tf.Tensor: shape=(), dtype=float32, numpy=0.007726725>,
#   <tf.Tensor: shape=(), dtype=float32, numpy=0.008077851>,
#   <tf.Tensor: shape=(), dtype=float32, numpy=0.011091873>,
#   <tf.Tensor: shape=(), dtype=float32, numpy=0.007991858>],
#  'Height+Amp Mod': [<tf.Tensor: shape=(), dtype=float32, numpy=0.0061029983>,
#   <tf.Tensor: shape=(), dtype=float32, numpy=0.0025726485>,
#   <tf.Tensor: shape=(), dtype=float32, numpy=0.0049142675>,
#   <tf.Tensor: shape=(), dtype=float32, numpy=0.005970302>,
#   <tf.Tensor: shape=(), dtype=float32, numpy=0.0026410087>,
#   <tf.Tensor: shape=(), dtype=float32, numpy=0.0046095704>,
#   <tf.Tensor: shape=(), dtype=float32, numpy=0.004720873>,
#   <tf.Tensor: shape=(), dtype=float32, numpy=0.0038892117>,
#   <tf.Tensor: shape=(), dtype=float32, numpy=0.0070206886>,
#   <tf.Tensor: shape=(), dtype=float32, numpy=0.0050272904>],
#  'Phase+Amp Mod': [<tf.Tensor: shape=(), dtype=float32, numpy=0.005253082>,
#   <tf.Tensor: shape=(), dtype=float32, numpy=0.004431437>,
#   <tf.Tensor: shape=(), dtype=float32, numpy=0.004634896>,
#   <tf.Tensor: shape=(), dtype=float32, numpy=0.005812933>,
#   <tf.Tensor: shape=(), dtype=float32, numpy=0.0045293607>,
#   <tf.Tensor: shape=(), dtype=float32, numpy=0.0062166015>,
#   <tf.Tensor: shape=(), dtype=float32, numpy=0.005635705>,
#   <tf.Tensor: shape=(), dtype=float32, numpy=0.0054054>,
#   <tf.Tensor: shape=(), dtype=float32, numpy=0.0077650147>,
#   <tf.Tensor: shape=(), dtype=float32, numpy=0.006980785>],
#  'Height+Phase Mod': [<tf.Tensor: shape=(), dtype=float32, numpy=0.010465389>,
#   <tf.Tensor: shape=(), dtype=float32, numpy=0.0060338555>,
#   <tf.Tensor: shape=(), dtype=float32, numpy=0.007930171>,
#   <tf.Tensor: shape=(), dtype=float32, numpy=0.009601837>,
#   <tf.Tensor: shape=(), dtype=float32, numpy=0.0053209853>,
#   <tf.Tensor: shape=(), dtype=float32, numpy=0.008865538>,
#   <tf.Tensor: shape=(), dtype=float32, numpy=0.0069896267>,
#   <tf.Tensor: shape=(), dtype=float32, numpy=0.0070503694>,
#   <tf.Tensor: shape=(), dtype=float32, numpy=0.011210598>,
#   <tf.Tensor: shape=(), dtype=float32, numpy=0.0077456348>],
#  'Height+Phase+Amp Mod': [<tf.Tensor: shape=(), dtype=float32, numpy=0.006888828>,
#   <tf.Tensor: shape=(), dtype=float32, numpy=0.0026324526>,
#   <tf.Tensor: shape=(), dtype=float32, numpy=0.006475952>,
#   <tf.Tensor: shape=(), dtype=float32, numpy=0.0065429723>,
#   <tf.Tensor: shape=(), dtype=float32, numpy=0.003009324>,
#   <tf.Tensor: shape=(), dtype=float32, numpy=0.005770042>,
#   <tf.Tensor: shape=(), dtype=float32, numpy=0.005289008>,
#   <tf.Tensor: shape=(), dtype=float32, numpy=0.004735568>,
#   <tf.Tensor: shape=(), dtype=float32, numpy=0.007541831>,
#   <tf.Tensor: shape=(), dtype=float32, numpy=0.0060184756>]}