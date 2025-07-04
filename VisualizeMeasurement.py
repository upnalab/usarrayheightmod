import pandas as pd
import os
import matplotlib.pyplot as plt
import ctypes
import numpy as np
from Waves import Waves

import matplotlib.lines as lines

#from picosdk.functions import adc2mV
#from picosdk.ps2000 import ps2000 as ps

plt.close('all')
emitterApperture = 0.009
c = 340
fr = 40000
wavelength = c / fr 
k = 2 * np.pi / wavelength


def plotFlipper(df,simCounterpart,wavelength):
    #We take the Experimental measurements' results
    df_Pa = df/df.max().max()
    np_Pa=df_Pa.to_numpy()
    np_Pa_Inv=np.flip(np_Pa, 0)
    
    
    #We take the simulation results   
    emPositions=np.array((simCounterpart.iloc[:,0],simCounterpart.iloc[:,1],simCounterpart.iloc[:,2])).T
    sizeXY=0.175
    resolution=64


    outputPositions = Waves.planeGridZ(0,0, 0.16, sizeXY, sizeXY, resolution, resolution)
    normals = Waves.constNormals(emPositions, [0,0,1])
    propagators = Waves.calcPropagatorsPistonsToPoints(emPositions, normals, outputPositions, k, emitterApperture)
    propagatorsConst =  propagators # Transform into a tf.constant to speed the training
    phasesColumn=np.array(simCounterpart.iloc[:,-1]).reshape(-1)
    emitters =   np.cos(phasesColumn)+ np.sin(phasesColumn) *1.0j
    field = emitters @ propagatorsConst
    ampField = np.abs( np.reshape(field, [64,64]) ) # -1 To be flatten
    ampFieldInv=np.flip(ampField, 0)
    
    #wavelenght calc for the subplot

    RealLifeSizePixel=sizeXY/resolution #sizeXY is the lenght of the simulated pattern, m
    realPixelsWavelength=wavelength/RealLifeSizePixel #wavelenght related to the image m/m 
    


    subplotLenght=5 #it is in inches, 1 inch= 2.54cm
    dpi=100

    figInchessWaveLenght=(subplotLenght/resolution)*realPixelsWavelength
    
    fig,ax= plt.subplots(ncols=2,figsize=(subplotLenght*2, subplotLenght),dpi=dpi)
    
    
    
    line_length_fig = figInchessWaveLenght / fig.get_size_inches()[0]  # Normalized width (0-1 range)    
    # Draw a line in figure coordinates
    x_start = 0.4  # Start position (normalized, 10% from the left edge)
    y_position = 0.2  # Position (normalized, 90% from the bottom)
    x_end = x_start + line_length_fig  # End position based on the desired length
    
    # Add the line using Line2D in figure coordinates
    line = lines.Line2D([x_start, x_end], [y_position, y_position], transform=fig.transFigure, color='red', lw=2)
    fig.add_artist(line)
    fig.text(x_start, y_position + 0.02, f'  λ', color='red')
    fig.suptitle(name) 
    ax[0].imshow(ampFieldInv, origin = 'lower')
    im=ax[1].imshow(np_Pa_Inv,origin = 'lower')
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.show()
    
    name2="FlipOption"+name
    fig2,ax2= plt.subplots(ncols=2,figsize=(subplotLenght*2, subplotLenght),dpi=dpi)
    line = lines.Line2D([x_start, x_end], [y_position, y_position], transform=fig2.transFigure, color='red', lw=2)
    fig2.add_artist(line)
    fig2.text(x_start, y_position + 0.02, f'  λ', color='red')
    fig2.suptitle(name2)
    ax2[0].imshow(ampFieldInv, origin = 'lower')
    im=ax2[1].imshow(np_Pa, origin = 'lower')
    fig2.subplots_adjust(right=0.8)
    cbar_ax = fig2.add_axes([0.85, 0.15, 0.03, 0.7])
    fig2.colorbar(im, cax=cbar_ax)
    
    name3="ExtraFlipOption"+name
    fig3,ax3= plt.subplots(ncols=2,figsize=(subplotLenght*2, subplotLenght),dpi=dpi)
    line = lines.Line2D([x_start, x_end], [y_position, y_position], transform=fig3.transFigure, color='red', lw=2)
    fig3.add_artist(line)
    fig3.text(x_start, y_position + 0.02, f'  λ', color='red')
    fig3.suptitle(name3)
    ax3[0].imshow(ampField, origin = 'lower')
    im=ax3[1].imshow(np_Pa, origin = 'lower')
    fig3.subplots_adjust(right=0.8)
    cbar_ax = fig3.add_axes([0.85, 0.15, 0.03, 0.7])
    fig3.colorbar(im, cax=cbar_ax)
    
    plt.show()
print(f'----HeightMod-----------')
for name in os.listdir('measurements/HeightMod'):
    print(name)
    
    if '.csv' in name:
        df = pd.read_csv(f'measurements/HeightMod/{name}', header = None)
        try:
            simCounterpart= pd.read_csv(f'measurements/Simulations/{name}', header = None)
            plotFlipper(df,simCounterpart,wavelength)
            
        except Exception as error:
            print(f'error: {error}')
            maxValue = 32767
            maxADC = ctypes.c_int16(maxValue)  # The oscilloscope will return data as ADC values,which range from -32767 to +32767. These need to be converted with adc2mV.
                                            # This function will accept the buffer_a, and output a list of equivalent values in mV
    
            # rangePico = ps.PS2000_VOLTAGE_RANGE['PS2000_10V']
            
            # # convert ADC counts data to mV
            # df_Pa = df
            # for i in df:
            #     for j in df:
            #         adc2mVChA = adc2mV([df.iloc[i,j]],
            #                            rangePico,
            #                            maxADC)
                    
            #         df_Pa.iloc[i,j] = adc2mVChA[0]
            
            # ratio = 0.5012
            # df_Pa = df_Pa * ratio
            
            df_Pa = df/df.max().max() 
            plt.figure()
            plt.imshow(df_Pa, origin = 'lower')#, cmap = "gist_heat")
            plt.title(name)
            plt.colorbar()
            plt.show()

            # name = "A_HeightMod_2"
            # df = pd.read_csv(f'measurements/{name}.csv', header = None)
            # plt.figure()
            # plt.imshow(df, origin = 'lower')
            # plt.title(name)
            # plt.show()
print(f'----PhasedArray-----------')
           
for name in os.listdir('measurements/PhasedArray'):
    print(name)
    if '.csv' in name:
        df = pd.read_csv(f'measurements/PhasedArray/{name}', header = None)
        try:
            simCounterpart= pd.read_csv(f'measurements/Simulations/{name}', header = None)
            plotFlipper(df,simCounterpart,wavelength)
            
        except Exception as error:
            print(f'error: {error}')
            maxValue = 32767
            maxADC = ctypes.c_int16(maxValue)  # The oscilloscope will return data as ADC values,which range from -32767 to +32767. These need to be converted with adc2mV.
                                            # This function will accept the buffer_a, and output a list of equivalent values in mV
    
            # rangePico = ps.PS2000_VOLTAGE_RANGE['PS2000_10V']
            
            # # convert ADC counts data to mV
            # df_Pa = df
            # for i in df:
            #     for j in df:
            #         adc2mVChA = adc2mV([df.iloc[i,j]],
            #                            rangePico,
            #                            maxADC)
                    
            #         df_Pa.iloc[i,j] = adc2mVChA[0]
            
            # ratio = 0.5012
            # df_Pa = df_Pa * ratio
            
            df_Pa = df/df.max().max() 
            plt.figure()
            plt.imshow(df_Pa, origin = 'lower')#, cmap = "gist_heat")
            plt.title(name)
            plt.colorbar()
            plt.show()
            
