

from utils import utilsGCode,utilsPico
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
 
### Main try ###
 
g = utilsGCode()
# Connect to the machine
g.connectToMachine(relativeMode = False)


squareSize = 175 # Maximum 175.36
x_center= 0 
y_center = 0
X_min = x_center - squareSize/2
X_max = x_center + squareSize/2
Y_min = y_center - squareSize/2
Y_max = y_center + squareSize/2
Z = 150 # 150 or 160
# El mejor patron de A con heightMod se obtuvo con Z = 160
 
N_stepX = 128 #32
N_stepY = 128 #32
 
values = np.zeros([N_stepX,N_stepY])
 
name = "12_PhasedArray_4" # Nombre ya cambiado (tambien se ha cambiado el range a 10V y lo de cojer 10 para evitar errores)

g.move(X=X_min, Y=Y_min, Z = Z)

try:
    # Connect to Pico
    mic = utilsPico()
    # Open PicoScope
    mic.openUnit()
    
    for posX, X in enumerate(np.linspace(X_min, X_max, num = N_stepX)):
        if posX % 2 == 0:
            RangeY = np.linspace(Y_min, Y_max, num = N_stepY)
        else:
            RangeY = np.linspace(Y_max, Y_min, num = N_stepY)
        for posY, Y in enumerate(RangeY):
                g.move(X = X, Y = Y, Z = Z)
                time.sleep(0.1)
                # Colllect data, add position and values to the lists and plot
                peak2peak, data = mic.collectData()
                if posX % 2 == 0:
                    values[posX,posY] = peak2peak
                else:
                    values[posX,len(RangeY)-1-posY] = peak2peak
                g.scatter(values, data)
                
    df = pd.DataFrame(values)  
    df.to_csv(f"measurements/{name}.csv", header = False, index = False)

except KeyboardInterrupt:
    pass
except Exception as e:
    print(e)
    pass

df = pd.DataFrame(values)  
df.to_csv(f"measurements/{name}.csv", header = False, index = False)

 
# Close all units
g.close()
mic.closeUnit()

