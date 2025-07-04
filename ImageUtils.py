#Load an image into an array
import tensorflow as tf
import numpy as np

class ImageUtils:
    def loadNorm(path, slicePx):
        img = tf.keras.preprocessing.image.load_img(path, color_mode='rgb')
        image_array  = tf.keras.utils.img_to_array(img)
        img = tf.image.rgb_to_grayscale(image_array)
        img = tf.image.resize(img, [slicePx,slicePx])
        target = tf.keras.utils.img_to_array(img)
        target = target / np.max( target )
        return target
    
    #TODO
    #def loadNormFromPath(dirPath, slicePx):
    