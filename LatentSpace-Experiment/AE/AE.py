import tensorflow as tf 
import numpy as np 
import datetime 
import os
import matplotlib.pyplot as plt 
from matplotlib import gridspec
from tensorflow.examples.tutorials.mnist import input_data


import argparse
	
mnist = input_data.read_data_sets('./Data',one_hot=True)

#parameters
input_dim = 784
n_l1,n_l2 = 1000,1000
z_dim = 2
batch_size = 100
nb_epoches = 200
learning_rate = 0.001
beta1 = 0.9
results_path = "./Results/Autoencoder"

x_input = tf.placeholder(dtype=tf.float32,shape=[batch_size,input_dim],name='Input')
x_target = tf.placeholder(dtype=tf.float32,shape=[batch_size,input_dim],name='Target')
decoder_input = tf.placeholder(dtype=tf.float32,shape=[1,z_dim],name='Decoder_input')


def generated_image_grid(sess,x_point,y_point,op):

	x_points = [x_point]
	y_points = [y_point]

	nx, ny = len(x_points), len(y_points)
	plt.subplot()
	gs = gridspec.GridSpec(nx, ny, hspace=0.05, wspace=0.05)

	for i, g in enumerate(gs):
		z = np.concatenate(([x_points[int(i / ny)]], [y_points[int(i % nx)]]))
		z = np.reshape(z, (1, 2))
		x = sess.run(op, feed_dict={decoder_input: z})
		ax = plt.subplot(g)
		img = np.array(x.tolist()).reshape(28, 28)
		ax.imshow(img, cmap='gray')
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_aspect('auto')
		# plt.show()
		name = "static/images/out.jpg"
		plt.savefig(name)
	return name
			


def form_results():
	folder_name = "/{0}_{1}_{2}_{3}_{4}_{5}_autoencoder".format(datetime.datetime.now(),z_dim,learning_rate,batch_size,nb_epoches,beta1)
	tensorboard_path = results_path+folder_name+"/Tensorboard"
	saved_model_path = results_path+folder_name+"/Saved_models/"
	log_path = results_path+folder_name+"/log"
	if not os.path.exists(results_path+folder_name):
		os.mkdir(results_path+folder_name)
		os.mkdir(tensorboard_path)
		os.mkdir(saved_model_path)
		os.mkdir(log_path)
	return tensorboard_path,saved_model_path,log_path


def dense(x,n1,n2,name):
	with tf.variable_scope(name,reuse=None):
		weights = tf.get_variable("weights",shape=[n1,n2],initializer=tf.random_normal_initializer(mean = 0.,stddev = 0.01))
		bias = tf.get_variable("bias",shape=[n2],initializer=tf.constant_initializer(0.0))
		out = tf.add(tf.matmul(x,weights),bias,name='matmul')
		return out



def encoder(x,reuse=False):
	if reuse:
		tf.get_variable_scope().reuse_variables()
	with tf.name_scope("Encoder"):
		e_dense_1 = tf.nn.relu(dense(x,input_dim,n_l1,name='e_dense_1'))
		e_dense_2 = tf.nn.relu(dense(e_dense_1,n_l1,n_l2,name='e_dense_2'))
		latent_variable = dense(e_dense_2,n_l2,z_dim,name='e_latent_variable')
		return latent_variable

def decoder(x,reuse=False):
	if reuse:
		tf.get_variable_scope().reuse_variables()
	with tf.name_scope('Decoder'):
		d_dense_1 = tf.nn.relu(dense(x,z_dim,n_l2,name='d_dense_1'))
		d_dense_2 = tf.nn.relu(dense(d_dense_1,n_l2,n_l1,name='d_dense_2'))
		output = tf.nn.sigmoid(dense(d_dense_2,n_l1,input_dim,name='output'))
		return output

def train(args):
	train_model = args.train_model
	x = int(args.x)
	y = int(args.y) 
	print("here ---------------------------")



	with tf.variable_scope(tf.get_variable_scope()):
		encode_output = encoder(x_input)
		decoder_output = decoder(encode_output)

	with tf.variable_scope(tf.get_variable_scope()):
		decoder_image = decoder(decoder_input,reuse=True)


	loss = tf.reduce_mean(tf.square(x_target - decoder_output))

	optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate,beta1=beta1).minimize(loss)
	init = tf.global_variables_initializer()

	tf.summary.scalar(name='Loss' , tensor = loss)
	tf.summary.histogram(name = 'Encoder Distrbution',values = encode_output)
	input_images = tf.reshape(x_input,[-1,28,28,1])
	generated_images = tf.reshape(decoder_output,[-1,28,28,1])
	tf.summary.image(name='Input Images',tensor=input_images,max_outputs = 10)
	tf.summary.image(name='Generated Images',tensor=generated_images,max_outputs=10)
	summary_op =tf.summary.merge_all()

	saver = tf.train.Saver()
	step = 0
	with tf.Session() as sess:
		sess.run(init)
		if train_model:
			tensorboard_path,saved_model_path,log_path = form_results()
			writer = tf.summary.FileWriter(logdir = tensorboard_path,graph=sess.graph)
			for i in range(nb_epoches):
				n_batches = int(mnist.train.num_examples / batch_size)
				for b in range(n_batches):
					batch_x,_ = mnist.train.next_batch(batch_size)
					sess.run(optimizer,feed_dict={x_input:batch_x,x_target:batch_x})
					if b % 50 ==0:
						batch_loss,summary = sess.run([loss,summary_op],feed_dict={x_input:batch_x,x_target:batch_x})
						writer.add_summary(summary,global_step=step)
						print("Loss: {}".format(batch_loss))
						print("Epoch:  {} , Iteration: {}".format(i,b))
						with open(log_path + '/log.txt','a') as log:
							log.write("Epoch: {},Iteration:{}\n".format(i,b))
							log.write("Loss: {}\n".format(batch_loss))

						step += 1

					saver.save(sess,save_path = saved_model_path,global_step=step)
			print("Model Trained")
			print("tensorboard_path: {}".format(tensorboard_path))
			print("log path : {}".format(log_path+'/log.txt'))
			print("Saved Model path: {}".format(saved_model_path))

		else:
			all_results = os.listdir(results_path)
			all_results.sort()
			saver.restore(sess,save_path = tf.train.latest_checkpoint(results_path+'/'+all_results[-1]+'/Saved_models/'))
			generated_image_grid(sess,x,y,op=decoder_image)




if __name__ == '__main__':
	a = argparse.ArgumentParser()
	a.add_argument("--x",default = "0")
	a.add_argument("--y",default = "0")
	a.add_argument("--train_model",default = False)
	args = a.parse_args()
	train(args)








