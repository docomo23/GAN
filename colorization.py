import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
import os
import scipy.misc
import scipy
from skimage import data_dir,io,transform,color
import numpy as np
import random


##### Helper function
def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)



class DCGAN():
    def __init__(self,image_shape,batch_size = 1,iterations = 500000):
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.iterations = iterations

        tf.reset_default_graph()


    # the architectures of generator and discriminator
    def generator(self,z):
        # This initializaer is used to initialize all the weights of the network.
        initializer = tf.truncated_normal_initializer(stddev=0.02)

        gen1 = slim.convolution2d_transpose(
            z, num_outputs=16, kernel_size=[10, 10], stride=[1, 1],
            padding="SAME", normalizer_fn=slim.batch_norm,
            activation_fn=lrelu, scope='g_conv1', weights_initializer=initializer)

        gen2 = slim.convolution2d_transpose(
            gen1, num_outputs=32, kernel_size=[5, 5], stride=[1, 1],
            padding="SAME", normalizer_fn=slim.batch_norm,
            activation_fn=lrelu, scope='g_conv2', weights_initializer=initializer)

        gen3 = slim.convolution2d_transpose(
            gen2, num_outputs=16, kernel_size=[3, 3], stride=[1, 1],
            padding="SAME", normalizer_fn=slim.batch_norm,
            activation_fn=lrelu, scope='g_conv3', weights_initializer=initializer)

        gen4 = slim.convolution2d_transpose(
            gen3, num_outputs=8, kernel_size=[2, 2], stride=[1, 1],
            padding="SAME", normalizer_fn=slim.batch_norm,
            activation_fn=lrelu, scope='g_conv4', weights_initializer=initializer)

        g_out = slim.convolution2d_transpose(
            gen4, num_outputs=3, kernel_size=[1, 1], stride=[1,1],padding="SAME",
            biases_initializer=None, activation_fn=lrelu,
            scope='g_out', weights_initializer=initializer)

        return g_out

    def discriminator(self,x, reuse=False):
        # This initializaer is used to initialize all the weights of the network.
        initializer = tf.truncated_normal_initializer(stddev=0.02)

        dis1 = slim.convolution2d(x, 16, [4, 4], stride=[2, 2], padding="SAME",
                                  biases_initializer=None, activation_fn=lrelu,
                                  reuse=reuse, scope='d_conv1', weights_initializer=initializer)

        dis2 = slim.convolution2d(dis1, 32, [4, 4], stride=[2, 2], padding="SAME",
                                  normalizer_fn=slim.batch_norm, activation_fn=lrelu,
                                  reuse=reuse, scope='d_conv2', weights_initializer=initializer)

        dis3 = slim.convolution2d(dis2, 64, [4, 4], stride=[2, 2], padding="SAME",
                                  normalizer_fn=slim.batch_norm, activation_fn=lrelu,
                                  reuse=reuse, scope='d_conv3', weights_initializer=initializer)

        d_out = slim.fully_connected(slim.flatten(dis3), 1, activation_fn=tf.nn.sigmoid,
                                     reuse=reuse, scope='d_out', weights_initializer=initializer)

        return d_out

    def train(self,sample_directory,model_directory):
        # These two placeholders are used for input into the generator and discriminator, respectively.
        z_in = tf.placeholder(shape=[None, self.image_shape[0],self.image_shape[1], 1], dtype=tf.float32)  # Random vector
        real_in = tf.placeholder(shape=[None, self.image_shape[0],self.image_shape[1], 3], dtype=tf.float32)  # Real images

        Gz = self.generator(z_in)  # Generates images from random z vectors
        Dx = self.discriminator(real_in)  # Produces probabilities for real images
        Dg = self.discriminator(Gz, reuse=True)  # Produces probabilities for generator images

        # These functions together define the optimization objective of the GAN.
        d_loss = -tf.reduce_mean(tf.log(Dx) + tf.log(1. - Dg))  # This optimizes the discriminator.
        g_loss = -tf.reduce_mean(tf.log(Dg))  # This optimizes the generator.

        tvars = tf.trainable_variables()



        # The below code is responsible for applying gradient descent to update the GAN.
        trainerD = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)
        trainerG = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)
        d_grads = trainerD.compute_gradients(d_loss,tvars[9:])  # Only update the weights for the discriminator network.
        g_grads = trainerG.compute_gradients(g_loss, tvars[0:9])  # Only update the weights for the generator network.

        update_D = trainerD.apply_gradients(d_grads)
        update_G = trainerG.apply_gradients(g_grads)
        init = tf.initialize_all_variables()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(init)
            for i in range(self.iterations):
                # sample from edge and color images
                zs, xs = self.sample_images()
                _, dLoss = sess.run([update_D, d_loss], feed_dict={z_in: zs, real_in: xs})  # Update the discriminator
                _, gLoss = sess.run([update_G, g_loss],
                                    feed_dict={z_in: zs})  # Update the generator, twice for good measure.
                _, gLoss = sess.run([update_G, g_loss], feed_dict={z_in: zs})
                if i % 10 == 0:
                    print(str(i)+"       "+str(gLoss) + "        " + str(dLoss))
                    z2,_ = self.sample_images(batch_size=1)
                    newZ = sess.run(Gz, feed_dict={z_in: z2})  # Use new z to get sample images from generator.
                    if not os.path.exists(sample_directory):
                        os.makedirs(sample_directory)
                    # Save sample generator images for viewing training progress.
                    io.imsave(arr=newZ[0],fname=sample_directory+'/fig'+str(i)+'.png')
                if i % 1000 == 0 and i != 0:
                    if not os.path.exists(model_directory):
                        os.makedirs(model_directory)
                    saver.save(sess, model_directory + '/model-' + str(i) + '.cptk')
                    print("Saved Model")

    # a helper function to sample images from directory
    def sample_images(self,batch_size=0):
        if batch_size == 0:
            batch_size = self.batch_size
        edge_images = []
        color_images = []
        indexes = random.sample(range(1,2),batch_size)
        edge_dir = '/Users/fengjiang/Desktop/projects/GAN/RoselatestEdge/'
        color_dir = '/Users/fengjiang/Desktop/projects/GAN/RoselatestResized/'
        for i in indexes:
            edge_images.append(np.expand_dims(io.imread(edge_dir+np.str(i)+'.jpg'),axis=2))
            color_images.append(np.array(io.imread(color_dir+np.str(i)+'.jpg')))
        return edge_images, color_images



def main():
    image_shape = [64,64]
    '''
    # Directory to save sample images from generator in.
    sample_directory = input('where to save sample images from generator (input example: ./figs ): ')
    # Directory to save trained model to.
    model_directory = input('where to save trained models (input example: ./models ): ')
    '''
    sample_directory = './color0507'
    model_directory='./color_model0507'
    battle = DCGAN(image_shape)
    battle.train(sample_directory,model_directory)



if __name__ == '__main__':
    main()
