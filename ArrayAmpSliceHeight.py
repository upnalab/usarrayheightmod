import tensorflow as tf
import numpy as np

class ArrayAmpSlice:
    def __init__(self):
        self.iters = 400
        self.showLossEvery = 20
        self.learningRate = 0.001
        
        self.c = 340
        self.fr = 40000
        self.wavelength = self.c / self.fr 
        
        
    def optimizeAmpSlice(self, target, distTarget, outputPositions, coordsXY, nEmitters, optimizeHeight = True, optimizePhase = False, optimizeAmp = False):        
        targetFlat = tf.constant( tf.reshape(target, [-1]) ) # To be flatten
        
        outputPositionsP = outputPositions[np.newaxis, :, :]
        coordsXY = coordsXY[:, np.newaxis, :]
        
        normals = [0,0,1]
        diffXY = outputPositionsP[:,:,:2] - coordsXY  

        #initial heights = 0
        if optimizeHeight:
            heights = tf.Variable( tf.zeros([nEmitters,1], dtype="float32"), trainable=True, 
                                  constraint= lambda h : tf.clip_by_value(h, clip_value_min = 0.0, clip_value_max = self.wavelength) ) # Min and Max value are totally arbitrary
        else:
            heights = tf.Variable( tf.zeros([nEmitters,1], dtype="float32"), trainable=False)
        
        if optimizePhase:
            phases = tf.Variable( tf.random.uniform([1,nEmitters], minval = -np.pi, maxval = np.pi, dtype="float32"), trainable=True,
                                  constraint= lambda h : tf.clip_by_value(h, clip_value_min = -np.pi, clip_value_max = np.pi) )
        
        else:
            phases = tf.Variable( tf.zeros([1,nEmitters], dtype="float32"), trainable=False )
            
        if optimizeAmp:
                cosAmps = tf.Variable( tf.random.uniform([1,nEmitters], minval=-np.pi, maxval=np.pi, dtype="float32"), trainable=True,
                                      constraint= lambda h : tf.clip_by_value(h, clip_value_min = -np.pi/2, clip_value_max = np.pi/2) )
        
        else:  
            cosAmps = tf.Variable( tf.zeros([1,nEmitters], dtype="float32"), trainable=False  )
        
        if not optimizeHeight:
            shapeA = diffXY.shape
            shapeB = outputPositionsP.shape
            propagators = np.zeros((shapeA[0], shapeB[0]), dtype=complex)
            diffZ = outputPositionsP[:,:,2] - heights
            
            nd = tf.sqrt(diffXY[:, :, 0]**2 + diffXY[:, :, 1]**2 + diffZ**2) 
            nn = np.linalg.norm(normals, axis=0)
    
            angle = np.arccos(
                (diffXY[:, :, 0] * normals[0]
                + diffXY[:, :,  1] * normals[1]
                + diffZ * normals[2]) / nd / (nn + 1e-07) # + 1e-07 to avoid division by zero
            )
            
            emitterApperture = 0.009
            c = 340
            fr = 40000
            wavelength = c / fr 
            k = 2 * np.pi / wavelength


            dum = 0.5 * emitterApperture * k * np.sin(angle)
            dire = np.sinc(dum / np.pi)
            propagators = tf.complex(dire / nd, 0.) * tf.complex(tf.cos(k * nd), tf.sin(k * nd))        
            propagatorsConst = tf.constant( propagators, dtype="complex64" ) # Transform into a tf.constant to speed the training
        else:
            propagatorsConst = None

        
        optim_height    = tf.keras.optimizers.Adam(learning_rate=self.learningRate)
        optim_phase     = tf.keras.optimizers.Adam(learning_rate=0.1)
        optim_amp       = tf.keras.optimizers.Adam(learning_rate=0.1)
            
        best_loss = 1
        best_heights = None
        best_phases = None
        best_amps = None
        for i in range(self.iters):
            with tf.GradientTape() as tape:
                loss = ArrayAmpSlice.targetFunction(heights, distTarget, outputPositionsP, diffXY, normals, targetFlat, phases, cosAmps, propagatorsConst, optimizeHeight)
                
                if loss < best_loss:
                    best_loss    = tf.convert_to_tensor(loss.numpy(),    dtype = tf.float32)
                    best_heights = tf.convert_to_tensor(heights.numpy(), dtype = tf.float32)
                    best_phases  = tf.convert_to_tensor(phases.numpy(),  dtype = tf.float32)
                    best_amps    = tf.convert_to_tensor(cosAmps.numpy(), dtype = tf.float32)
                
                    
                if optimizeHeight and not optimizePhase and not optimizeAmp:
                    grads = tape.gradient(loss, [heights])
                    optim_height.apply_gradients(zip(grads, [heights]))
                
                if not optimizeHeight and optimizePhase and not optimizeAmp:
                    grads = tape.gradient(loss, [phases])
                    optim_phase.apply_gradients(zip(grads, [phases]))
                
                if optimizeHeight and not optimizePhase and optimizeAmp:
                    grads = tape.gradient(loss, [heights, cosAmps])
                    optim_height.apply_gradients([(grads[0], heights)])
                    optim_amp.apply_gradients([(grads[1], cosAmps)])
                    
                if not optimizeHeight and optimizePhase and optimizeAmp:
                    grads = tape.gradient(loss, [phases, cosAmps])                    
                    optim_phase.apply_gradients([(grads[0], phases)])
                    optim_amp.apply_gradients([(grads[1], cosAmps)])
                    
                if optimizeHeight and optimizePhase and not optimizeAmp:
                    grads = tape.gradient(loss, [heights, phases])
                    optim_height.apply_gradients([(grads[0], heights)])
                    optim_phase.apply_gradients([(grads[1], phases)])
                
                if optimizeHeight and optimizePhase and optimizeAmp:
                    grads = tape.gradient(loss, [heights, phases, cosAmps])
                    optim_height.apply_gradients([(grads[0], heights)])
                    optim_phase.apply_gradients([(grads[1], phases)])
                    optim_amp.apply_gradients([(grads[2], cosAmps)])
            
            if i % self.showLossEvery == 0 and i != 0:
                print(f'Iteration {i} of {self.iters} Loss = {loss}')
                
                
        ampField = ArrayAmpSlice.outputField(best_heights, distTarget, outputPositionsP, diffXY, normals, best_phases, best_amps, propagatorsConst, optimizeHeight)
        ampField = tf.reshape( ampField, [target.shape[0], target.shape[1]])
        
        emitterPositions = np.concatenate((coordsXY[:,0,:], np.array([heights[:,0]]).T), axis=1)
        return best_loss, best_heights, best_phases, emitterPositions, ampField, tf.cos( best_amps )
    
    def outputField(heights, distTarget, outputPositionsP, diffXY, normals, phases, cosAmps, propagatorsConst, heightMod):
        if heightMod:
            shapeA = diffXY.shape
            shapeB = outputPositionsP.shape
            propagators = np.zeros((shapeA[0], shapeB[0]), dtype=complex)
            
            diffZ = outputPositionsP[:,:,2] - heights
            
            nd = tf.sqrt(diffXY[:, :, 0]**2 + diffXY[:, :, 1]**2 + diffZ**2) 
            nn = np.linalg.norm(normals, axis=0)
    
            angle = np.arccos(
                (diffXY[:, :, 0] * normals[0]
                + diffXY[:, :,  1] * normals[1]
                + diffZ * normals[2]) / nd / (nn + 1e-07) # + 1e-07 to avoid division by zero
            )
            
            emitterApperture = 0.009
            c = 340
            fr = 40000
            wavelength = c / fr 
            k = 2 * np.pi / wavelength

            dum = 0.5 * emitterApperture * k * np.sin(angle)
            dire = np.sinc(dum / np.pi)
            propagators = tf.complex(dire / nd, 0.) * tf.complex(tf.cos(k * nd), tf.sin(k * nd)) 
            propagatorsConst = tf.constant( propagators, dtype="complex64" ) # Transform into a tf.constant to speed the training
            
        # All emitters Amplitude 1 and phase = phases
        amps = tf.cos( cosAmps )
        emitters =  tf.complex(amps * tf.cos(phases), amps * tf.sin(phases) )
        
        field = emitters @ propagatorsConst
        
        ampField = tf.abs( tf.reshape(field, [-1]) ) # -1 To be flatten
        return ampField
            
    def targetFunction(heights, distTarget, outputPositionsP, diffXY, normal, targetFlat, phases, cosAmps, propagatorsConst, heightMod):
        ampField = ArrayAmpSlice.outputField(heights, distTarget, outputPositionsP, diffXY, normal, phases, cosAmps, propagatorsConst, heightMod)
        
        maxAmp = tf.reduce_max( ampField )
        ampField = ampField / maxAmp
        loss = tf.reduce_mean(tf.square(ampField - targetFlat)) #mse
        
        return loss

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from ImageUtils import ImageUtils
    from Waves import Waves
    
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
    cPath = "patterns/"
    path = "A" # Example pattern
    target = ImageUtils.loadNorm(cPath + path + ".png", slicePx)
    
    lX = np.linspace(0 - arraySize / 2, 0 + arraySize / 2, emittersPerSide)
    lY = np.linspace(0 - arraySize / 2, 0 + arraySize / 2, emittersPerSide)
    coordsXY = np.meshgrid(lX, lY)
    coordsXY = np.vstack([np.ravel(coord) for coord in coordsXY]).T
    
    outputPositions = Waves.planeGridZ(0,0, distTarget, targetSize, targetSize, slicePx, slicePx)
    
    
    opti = ArrayAmpSlice()
    opti.iters = 1500
    opti.showLossEvery = 100 #opti.iters + 1
    loss, heights, phases, emitterPositions, ampField, amps = opti.optimizeAmpSlice(target, distTarget, outputPositions, coordsXY, nEmitters)
    
    # Plot the example
    plt.figure()
    plt.imshow(ampField, cmap = plt.cm.gist_heat, extent=[outputPositions[:,0].min(),outputPositions[:,0].max(),outputPositions[:,1].min(),outputPositions[:,1].max()])
    plt.colorbar(label = "Normalized Amplitude")
    plt.xlabel('x(mm)')
    plt.ylabel('y(mm)')
    plt.show()
    
    # Plot the example in 3D
    ax = plt.figure().add_subplot(projection='3d')
    cs2 = ax.scatter(emitterPositions[:,0], emitterPositions[:,1], emitterPositions[:,2], c = phases, alpha = np.round(amps,5), cmap = plt.cm.hsv, vmin = -np.pi, vmax = np.pi)
    cs  = ax.scatter(outputPositions[:,0],  outputPositions[:,1],  outputPositions[:,2], c = ampField, cmap = plt.cm.gist_heat)
    cbar = plt.colorbar(cs2, ax = ax, label='Phase (rad)')
    ticks = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
    tick_labels = [r"$-\pi$", r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$", r"$\pi$"]

    cbar.set_ticks(ticks)
    cbar.set_ticklabels(tick_labels)
    
    ax.set_zticks(np.arange(0, distTarget+1e-2, 2e-2))
    ax.set_xticks(np.arange(outputPositions[:,0].min(),outputPositions[:,0].max()+1e-2, 2e-2))
    ax.set_yticks(np.arange(outputPositions[:,1].min(),outputPositions[:,1].max()+1e-2, 2e-2))
    plt.show()
        
        

    
    


    