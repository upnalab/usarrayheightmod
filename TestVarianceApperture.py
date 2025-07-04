import pandas as pd
import numpy as np
from Waves import Waves
from ImageUtils import ImageUtils
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import tensorflow as tf

# File format: x,y,z,nx,ny,nz,power,frequency,apperture,Type(0=circle,1=square),sx,sy,sz,phase
file = "heightMod_05_06_2024_14_00_20.csv"
fileCsv = pd.read_csv('Results/' + file, delimiter=',', header = None, names = ['x','y','z','nx','ny','nz','power','frequency','apperture','type','sx','sy','sz','phase'])


targetSize = 0.16
slicePx = 128
distTarget = 0.16
c = 340
fr = fileCsv['frequency'][0]
wavelength = c / fr 
k = 2 * np.pi / wavelength
emitterApperture = fileCsv['apperture']

emitterPositions = np.vstack([fileCsv['x'], -fileCsv['y'], fileCsv['z']]).T
normals = Waves.constNormals(emitterPositions, [fileCsv['nx'][0],fileCsv['ny'][0],fileCsv['nz'][0]])
outputPositions = Waves.planeGridZ(0,0, distTarget, targetSize, targetSize, slicePx, slicePx)

# Target
target = ImageUtils.loadNorm("patterns/thickA.png", slicePx)
targetFlat = tf.constant( tf.reshape(target, [-1]) ) # To be flatten

def compute_differentSTD(std):
    # Add gaussian noise
    mean = 1
    noise = np.random.normal(mean, std, size = emitterApperture.shape)
    emitterAppertureNoise = emitterApperture * noise
    
    props = Waves.calcPropagatorsPistonsToPoints(emitterPositions, normals, outputPositions, k, emitterAppertureNoise, difApperture = True)
    
    
    amps = np.ones(emitterPositions.shape[0])
    phases = fileCsv['phase']
    emission = tf.complex(amps * tf.cos(phases), amps * tf.sin(phases) ) # shape (emitterPositions.shape[0],) we want it to have shape (1, emitterPositions.shape[0])
    emission = tf.reshape(emission, [1, emission.shape[0]])
    
    
    field = emission @ props 
    fieldAmp = np.abs(field)
    fieldAmpNorm = fieldAmp / np.max(fieldAmp)
    
    loss = tf.reduce_mean(tf.square(fieldAmpNorm - targetFlat)) #mse
    fieldAmpNormReshaped = np.reshape(fieldAmpNorm, [slicePx, slicePx])
    return loss, fieldAmpNormReshaped


# Create the figure and the line that we will manipulate
fig, ax = plt.subplots()
loss, fieldAmpNormReshaped = compute_differentSTD(std = 0) # No deviation at first
ax.imshow(fieldAmpNormReshaped, cmap = "gist_heat")
ax.set_title(f'Mse: {loss}')

# adjust the main plot to make room for the sliders
fig.subplots_adjust(left=0.25, bottom=0.25)

# Make a horizontal slider to control the frequency.
axstd = fig.add_axes([0.25, 0.1, 0.65, 0.03])
std_slider = Slider(
    ax=axstd,
    label='Std',
    valmin=0,
    valmax=2,
    valinit=0,
)

# The function to be called anytime a slider's value changes
def update(val):
    ax.cla()
    
    loss, fieldAmpNormReshaped = compute_differentSTD(std = val)
    ax.imshow(fieldAmpNormReshaped, cmap = "gist_heat")
    ax.set_title(f'Mse: {loss}')

    
    fig.canvas.draw_idle()


# register the update function with each slider
std_slider.on_changed(update)

plt.show()

