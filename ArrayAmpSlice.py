import tensorflow as tf
import numpy as np


class ArrayAmpSlice:
    def __init__(self):
        self.iters = 1000
        self.showLossEvery = 20
        self.learningRate = 0.016
        self.normalizeOutputAmp = True
        self.c = 340
        self.fr = 40000
        self.wavelength = self.c / self.fr
        self.useSsim=0.1

        
    def optimizeAmpSlice(self, target, distTarget, outputPositions, coordsXY, nEmitters):        
        targetFlat = tf.constant( tf.reshape(target, [-1]) ) # To be flatten
        
        outputPositionsP = outputPositions[np.newaxis, :, :]
        coordsXY = coordsXY[:, np.newaxis, :]
        
        normals = [0,0,1]
        diffXY = outputPositionsP[:,:,:2] - coordsXY 
        slicePx=int(np.sqrt(outputPositions.shape[0]))
        
        print(slicePx)

        #initial heights = 0
        heights = tf.Variable( tf.zeros([nEmitters,1], dtype="float32"), trainable=False, 
                              constraint= lambda h : tf.clip_by_value(h, clip_value_min = 0., clip_value_max = self.wavelength) ) # Min and Max value are totally arbitrary
        phases = tf.Variable( tf.zeros([1,nEmitters], dtype="float32"), trainable=True )
        cosAmps = tf.Variable( tf.zeros([1,nEmitters], dtype="float32"), trainable=False  )
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
        # optimizer = tf.keras.optimizers.Adam(learning_rate=self.learningRate)
        
        # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        #     self.learningRate,
        #     decay_steps=10,  
        #     decay_rate=0.96,
        #     staircase=True
        # )
        # optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=LRSchedule(self.learningRate))
        
        for i in range(self.iters):
            with tf.GradientTape() as tape:
                # loss = ArrayAmpSlice.targetFunction(heights, distTarget, outputPositionsP, diffXY, normals, targetFlat, self.normalizeOutputAmp, phases, cosAmps)
                loss = self.targetFunction(cosAmps,phases,propagatorsConst,targetFlat,self.normalizeOutputAmp,1,self.useSsim,slicePx)
                
                grads = tape.gradient(loss, [phases])
                optimizer.apply_gradients(zip(grads, [phases]))
            
        
            if i % self.showLossEvery == 0:
                print(f'Iteration {i} of {self.iters} Lr = {optimizer.learning_rate(optimizer.iterations)} Loss = {loss}')
                 
                
        ampField = ArrayAmpSlice.outputField(cosAmps, phases, propagators, 1)
        maxAmp = tf.reduce_max( ampField )
        ampField = ampField / maxAmp
        loss = tf.reduce_mean(tf.square(ampField - targetFlat)) #mse
        ampField = tf.reshape( ampField, [target.shape[0], target.shape[1]])
        emitterPositions = np.concatenate((coordsXY[:,0,:], np.array([heights[:,0]]).T), axis=1)
        print(f'final MSE:{loss}')
        
        return loss.numpy(), heights, phases, emitterPositions, ampField, cosAmps
    
    
    def targetFunction(self,cosAmps, phases, propagators, targetFlat, normalizeOutput, nMux, useSsim, slicePx, externalPlate=None):
        ampField = ArrayAmpSlice.outputField(cosAmps, phases, propagators, nMux,externalPlate)
        if normalizeOutput:
            maxAmp = tf.reduce_max( ampField )
            ampField = ampField / maxAmp
        
        if self.useSsim == 0:
            loss = tf.reduce_mean(tf.square(ampField - targetFlat)) #mse  
           
        elif self.useSsim == 1:
 
            img1 = tf.expand_dims(tf.reshape(ampField, [slicePx,slicePx]), axis = 2)
            img2 = tf.expand_dims(tf.reshape(targetFlat, [slicePx,slicePx]), axis = 2)
            ssim = 1 - tf.image.ssim(img1, img2, max_val=1) # Default values are filter_size=11,filter_sigma=1.5,k1=0.01,k2=0.03,return_index_map=False)
            loss = ssim
        else:
            mse = tf.reduce_mean(tf.square(ampField - targetFlat)) #mse
            # Let's add the structural similarity as a metric to the loss (It must be use only when normalizeOutput)
            img1 = tf.expand_dims(tf.reshape(ampField, [slicePx,slicePx]), axis = 2)
            img2 = tf.expand_dims(tf.reshape(targetFlat, [slicePx,slicePx]), axis = 2)
            ssim = 1 - tf.image.ssim(img1, img2, max_val=1) # Default values are filter_size=11,filter_sigma=1.5,k1=0.01,k2=0.03,return_index_map=False)
            # NOTE: ssim goes from 0 to 1, where 1 means the ssim is perfect so in order to keep minimazing, because we want to
            # minimize the mse we must put 1 - ssim
            loss =(1-self.useSsim)* mse + self.useSsim*ssim
           
        return loss
    
    # def outputField(heights, distTarget, outputPositionsP, diffXY, normals, phases, cosAmps):
    #     shapeA = diffXY.shape
    #     shapeB = outputPositionsP.shape
    #     propagators = np.zeros((shapeA[0], shapeB[0]), dtype=complex)
        
    #     diffZ = outputPositionsP[:,:,2] - heights
        
    #     nd = tf.sqrt(diffXY[:, :, 0]**2 + diffXY[:, :, 1]**2 + diffZ**2) 
    #     nn = np.linalg.norm(normals, axis=0)

    #     angle = np.arccos(
    #         (diffXY[:, :, 0] * normals[0]
    #         + diffXY[:, :,  1] * normals[1]
    #         + diffZ * normals[2]) / nd / (nn + 1e-07) # + 1e-07 to avoid division by zero
    #     )
        
    #     emitterApperture = 0.009
    #     c = 340
    #     fr = 40000
    #     wavelength = c / fr 
    #     k = 2 * np.pi / wavelength

    #     dum = 0.5 * emitterApperture * k * np.sin(angle)
    #     dire = np.sinc(dum / np.pi)
    #     propagators = tf.complex(dire / nd, 0.) * tf.complex(tf.cos(k * nd), tf.sin(k * nd))        
    #     propagatorsConst = tf.constant( propagators, dtype="complex64" ) # Transform into a tf.constant to speed the training
        
    #     # All emitters Amplitude 1 and phase = phases
    #     amps = tf.cos( cosAmps )
    #     emitters =  tf.complex(amps * tf.cos(phases), amps * tf.sin(phases) )
        
    #     field = emitters @ propagatorsConst
        
    #     ampField = tf.abs( tf.reshape(field, [-1]) ) # -1 To be flatten
    #     return ampField
            
    # def targetFunction(heights, distTarget, outputPositionsP, diffXY, normal, targetFlat, normalizeOutputAmp, phases, cosAmps):
    #     ampField = ArrayAmpSlice.outputField(heights, distTarget, outputPositionsP, diffXY, normal, phases, cosAmps)
        
    #     if normalizeOutputAmp:
    #         maxAmp = tf.reduce_max( ampField )
    #         ampField = ampField / maxAmp
    #     loss = tf.reduce_mean(tf.square(ampField - targetFlat)) #mse
        
    #     return loss
    
    def outputField(cosAmps, phases, propagators, nMux,externalPlate=None):
        amps = tf.cos( cosAmps )
        emitters =  tf.complex(amps * tf.cos(phases), amps * tf.sin(phases) )
        if nMux == 1:
            field = emitters @ propagators
            if  externalPlate is not None:
                fieldWithPlate= field @ externalPlate
                ampField = tf.abs( fieldWithPlate )
            else:
                ampField = tf.abs( field )
            return ampField
        else:
            nFieldPoints = propagators.shape[1]
            ampField = tf.zeros( [ nFieldPoints ] )
            for iField in range(nMux):
                field = emitters[iField:iField+1, :] @ propagators
                fieldAmp = tf.abs(field)
                ampField = ampField + fieldAmp
            return ampField / nMux
        
    

class LRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, initial_learning_rate):
        self.learning_rate = initial_learning_rate
        self.decay_steps=10  
        self.decay_rate=0.999
        
    def __call__(self, step):
        # If step is less that 500 then decay the learning rate
        if step > 300 and step < 3000:
            if step.numpy() % self.decay_steps == 0 and step != 0:
                self.learning_rate *= self.decay_rate

        # Else remain stable so it does not go too low
        
        return self.learning_rate



    