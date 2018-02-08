"""
"""
import argparse

import tensorflow as tf
import numpy as np
import vgg19Base as vgg19
import utils, utils2

def loadImageData(path, labelNumber):
    [_, imagesFullpaths] = utils2.get_files_in_dir(path)
    images = []
    for imagePath in imagesFullpaths:
        print(imagePath)
        img = utils.load_image(imagePath)
        img.reshape((224, 224, 3))
        images.append(img)

    allImages = np.array(images)
    return (allImages, imagesFullpaths)

def testNetwork(images, modelFile, classFile, oriImages):
    with tf.device('/cpu:0'):
        sess = tf.Session()
        
        inputs = tf.placeholder(tf.float32, [None, 224, 224, 3])
        true_out = tf.placeholder(tf.float32, [None, 10])
        train_mode = tf.placeholder(tf.bool)
        
        vgg = vgg19.Vgg19(modelFile)
        vgg.build(inputs, 10, train_mode)
        
        # print number of variables used: 143667240 variables, i.e. ideal size = 548MB
        print("total variables in the trained model: " + str(vgg.get_var_count()))
        
        sess.run(tf.global_variables_initializer())
        
        # test classification
        probs = sess.run(vgg.prob, feed_dict={inputs: images, train_mode: False})
        index = 0
        for prob in probs:
            print("image = " + str(oriImages[index]))
            utils.print_prob(prob, classFile)
            index = index + 1
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Predict image class with trained model')
    parser.add_argument('-p', help = 'The path of images.')
    parser.add_argument('-s', help = 'The class file.')
    parser.add_argument('-m', help = 'The model filename.')

    args = parser.parse_args()
    if (args.p != None and args.s != None and args.m != None):
        [inputs, oriInputs] = loadImageData(args.p, 10)
        testNetwork(inputs, args.m, args.s, oriInputs)
    else:
        print("Usage: python predict.py")
