import pandas as pd
import numpy as np
import solid2
import os
import sys
sys.path.append('..')

cPath = "measurements/Simulations/"
for file in os.listdir(cPath):
    
    if not '.csv' in file:
        continue
    
    emitters = pd.read_csv(cPath + file, header = None)
    emitters = np.asarray(emitters.iloc[:, :3]) * 1000 # Because scad default units is milimeters, so we have to conver our units to mm
    
    baseHeight = 0.2
    widthHeight = abs(emitters[0,0] - emitters[1,0])
    
    for index, em in enumerate(emitters):
        if index == 0:
            mesh = solid2.translate((em[0] - widthHeight/2 ,em[1] - widthHeight/2 , 0))(
                    solid2.cube((widthHeight, widthHeight, em[2] + baseHeight))
                )
        else:
            mesh += solid2.translate((em[0] - widthHeight/2 ,em[1] - widthHeight/2 , 0))(
                    solid2.cube((widthHeight, widthHeight, em[2] + baseHeight))
                )
    
    mesh.save_as_scad(f'results/meshes/Mesh_{file.split(".")[0]}.scad')