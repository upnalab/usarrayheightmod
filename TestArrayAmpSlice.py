import numpy as np
from Waves import Waves
from ImageUtils import ImageUtils
from ArrayAmpSlice import ArrayAmpSlice
import open3d as o3d
from datetime import datetime
import pandas as pd 
import copy


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

#targets
target = ImageUtils.loadNorm("patterns/Star.png", slicePx)

lX = np.linspace(0 - arraySize / 2, 0 + arraySize / 2, emittersPerSide)
lY = np.linspace(0 - arraySize / 2, 0 + arraySize / 2, emittersPerSide)
coordsXY = np.meshgrid(lX, lY)
coordsXY = np.vstack([np.ravel(coord) for coord in coordsXY]).T

outputPositions = Waves.planeGridZ(0,0, distTarget, targetSize, targetSize, slicePx, slicePx)


opti = ArrayAmpSlice()
opti.iters = 9000
loss, heights, phases, emitterPositions, ampField, cosAmps = opti.optimizeAmpSlice(target, distTarget, outputPositions, coordsXY, nEmitters)


list_to_show = []
for em in range(len(emitterPositions)):
    # emitterStl = o3d.io.read_triangle_mesh("Transductor_10mm.stl")
    
    height_cyl = 7
    cylinder_main = o3d.geometry.TriangleMesh.create_cylinder(radius = 9.8/2, height = height_cyl, resolution = 20)
    cylinder_stick1 = o3d.geometry.TriangleMesh.create_cylinder(radius = 0.7/2, height = height_cyl, resolution = 10)
    cylinder_stick2 = o3d.geometry.TriangleMesh.create_cylinder(radius = 0.7/2, height = height_cyl, resolution = 10)

    cylinder_stick1 = cylinder_stick1.compute_vertex_normals()
    cylinder_stick1_vertices = np.asarray(cylinder_stick1.vertices)
    cylinder_stick1_vertices = cylinder_stick1_vertices + np.array([0, 2.5, -height_cyl])
    cylinder_stick1.vertices = o3d.utility.Vector3dVector(cylinder_stick1_vertices)

    cylinder_stick2 = cylinder_stick2.compute_vertex_normals()
    cylinder_stick2_vertices = np.asarray(cylinder_stick2.vertices)
    cylinder_stick2_vertices = cylinder_stick2_vertices + np.array([0, -2.5, -height_cyl])
    cylinder_stick2.vertices = o3d.utility.Vector3dVector(cylinder_stick2_vertices)

    emitterStl = cylinder_main + cylinder_stick1 + cylinder_stick2
    
    
    emitterStl = emitterStl.compute_vertex_normals()
    emitterStl_vertices = np.asarray(emitterStl.vertices) * 0.001


    vertices_circleEm = emitterStl_vertices[emitterStl_vertices[:,2] == emitterStl_vertices[:,2].max()]
    center_em = vertices_circleEm.mean(axis = 0)
    
    emitterStl_vertices = emitterStl_vertices + ( emitterPositions[em] - center_em)
    
    emitterStl.vertices = o3d.utility.Vector3dVector(emitterStl_vertices)
    
    list_to_show.append(emitterStl)
        

colors = np.zeros(outputPositions.shape)
colorsEmitters=np.zeros([emitterPositions.shape[0],3])
colorsEmitters[:,1]=1
ampField = np.array(ampField).flatten()
colors[:,0] = ampField/ampField.max()

emitters_cloud = o3d.geometry.PointCloud()
emitters_cloud.points = o3d.utility.Vector3dVector(emitterPositions)
emitters_cloud.colors = o3d.utility.Vector3dVector(colorsEmitters)
list_to_show.append(emitters_cloud)

outputPositions_cloud = o3d.geometry.PointCloud()
outputPositions_cloud.points = o3d.utility.Vector3dVector(outputPositions)
outputPositions_cloud.colors = o3d.utility.Vector3dVector(colors)
list_to_show.append(outputPositions_cloud)


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

# datetime
date = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

df.to_csv(f'results/heightMod_{date}.csv', sep = ',', header = False, index = False)


# 3D Visualization of the result
list_to_show2 = copy.deepcopy(list_to_show)
combined_mesh = list_to_show2[0] 
for i in range(1, len(list_to_show2[:-2])):
    combined_mesh += list_to_show2[i]

# combined_mesh.scale(1000, center = combined_mesh.get_center())
o3d.io.write_triangle_mesh(f"results/output_mesh_{date}.stl", combined_mesh)

o3d.visualization.draw_geometries(list_to_show)