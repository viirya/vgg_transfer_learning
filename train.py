"""
"""
import argparse

import tensorflow as tf
import numpy as np
import vgg19Base as vgg19
import utils, utils2

def loadTrainingData(imagesPath, path):
    inputs = [i.split() for i in utils2.load_file(path)]
    images = []
    labels = []
    for oneInput in inputs:
        imagePath = imagesPath + '/' + str(oneInput[0])
        img = utils.load_image(imagePath)
        img.reshape((224, 224, 3))
        images.append(img)

        label = [1 if i == int(oneInput[1]) else 0 for i in range(10)]
        labels.append(label)
    allImages = np.array(images)
    allLabels = np.array(labels)
    return (allImages, allLabels)

def testNetwork(images, labels, classFile, saveModelName):
    with tf.device('/cpu:0'):
        sess = tf.Session()
        
        inputs = tf.placeholder(tf.float32, [None, 224, 224, 3])
        true_out = tf.placeholder(tf.float32, [None, 10])
        train_mode = tf.placeholder(tf.bool)
        
        vgg = vgg19.Vgg19('./vgg19.npy')
        vgg.build(inputs, 10, train_mode)
        
        # print number of variables used: 143667240 variables, i.e. ideal size = 548MB
        print("total variables in the model: " + str(vgg.get_var_count()))
        
        sess.run(tf.global_variables_initializer())
        
        # test classification
        probs = sess.run(vgg.prob, feed_dict={inputs: images, train_mode: False})
        for prob in probs:
            utils.print_prob(prob, classFile)
        
        for i in range(20):
            print("training iter = " + str(i))

            cost = tf.reduce_sum((vgg.prob - true_out) ** 2)
            train = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)
            sess.run(train, feed_dict={inputs: images, true_out: labels, train_mode: True})
        
            # test classification again, should have a higher probability about tiger
            probs = sess.run(vgg.prob, feed_dict={inputs: images, train_mode: False})
            for prob in probs:
                utils.print_prob(prob, classFile)
        
        # test save
        if saveModelName != None:
            print("Saving trained model to " + saveModelName)
            vgg.save_npy(sess, saveModelName)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Transfer learning on VGG19')
    parser.add_argument('-p', help = 'The path of images.')
    parser.add_argument('-t', help = 'The training input file.')
    parser.add_argument('-s', help = 'The class file.')
    parser.add_argument('-o', help = 'The output model filename.')

    args = parser.parse_args()
    if (args.t != None and args.p != None and args.s != None):
        [inputs, labels] = loadTrainingData(args.p, args.t)
        testNetwork(inputs, labels, args.s, args.o)
    else:
        print("Usage: python train.py -p path_to_images -t training_input -s class_file [-o output_model]")
