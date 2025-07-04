import tensorflow as tf
import numpy as np


class ArrayAmpSliceHeight:
    def __init__(self):
        self.iters = 500
        self.showLossEvery = 20
        self.learningRate = 0.001
        self.normalizeOutputAmp = True
        
    def optimizeAmpSlice(self, target, distTarget, outputPositions, coordsXY, nEmitters):        
        targetFlat = tf.constant( tf.reshape(target, [-1]) ) # To be flatten
        
        outputPositionsP = outputPositions[np.newaxis, :, :]
        coordsXY = coordsXY[:, np.newaxis, :]
        
        normals = [0,0,1]
        diffXY = outputPositionsP[:,:,:2] - coordsXY  

        #initial heights = 0
        heights = tf.Variable( tf.zeros([nEmitters,1], dtype="float32"), trainable=True, 
                              constraint= lambda h : tf.clip_by_value(h, clip_value_min = 0., clip_value_max = distTarget/2) ) # Min and Max value are totally arbitrary
        phases = tf.Variable( tf.zeros([1,nEmitters], dtype="float32"), trainable=False )
        cosAmps = tf.Variable( tf.zeros([1,nEmitters], dtype="float32"), trainable=False  )

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learningRate)
        for i in range(self.iters):
            with tf.GradientTape() as tape:
                loss = ArrayAmpSliceHeight.targetFunction(heights, distTarget, outputPositionsP, diffXY, normals, targetFlat, self.normalizeOutputAmp, phases, cosAmps)
                
                grads = tape.gradient(loss, [heights])
                optimizer.apply_gradients(zip(grads, [heights]))
            
        
            if (i+1) % self.showLossEvery == 0:
                print(f'Iteration {i} of {self.iters} Loss = {loss}')
                
        ampField = ArrayAmpSliceHeight.outputField(heights, distTarget, outputPositionsP, diffXY, normals, phases, cosAmps)
        ampField = tf.reshape( ampField, [target.shape[0], target.shape[1]])
        
        emitterPositions = np.concatenate((coordsXY[:,0,:], np.array([heights[:,0]]).T), axis=1)
        return loss.numpy(), heights, phases, emitterPositions, ampField, cosAmps
    
    def outputField(heights, distTarget, outputPositionsP, diffXY, normals, phases, cosAmps):
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
            
    def targetFunction(heights, distTarget, outputPositionsP, diffXY, normal, targetFlat, normalizeOutputAmp, phases, cosAmps):
        ampField = ArrayAmpSliceHeight.outputField(heights, distTarget, outputPositionsP, diffXY, normal, phases, cosAmps)
        
        if normalizeOutputAmp:
            maxAmp = tf.reduce_max( ampField )
            ampField = ampField / maxAmp
        loss = tf.reduce_mean(tf.square(ampField - targetFlat)) #mse
        
        return loss


    