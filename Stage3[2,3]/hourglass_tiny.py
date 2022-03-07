# -*- coding: utf-8 -*-
"""
Deep Human Pose Estimation

Project by Walid Benbihi
MSc Individual Project
Imperial College
Created on Mon Jul 10 19:13:56 2017

@author: Walid Benbihi
@mail : w.benbihi(at)gmail.com
@github : https://github.com/wbenbihi/hourglasstensorlfow/

Abstract:
	This python code creates a Stacked Hourglass Model
	(Credits : A.Newell et al.)
	(Paper : https://arxiv.org/abs/1603.06937)
	
	Code translated from 'anewell' github
	Torch7(LUA) --> TensorFlow(PYTHON)
	(Code : https://github.com/anewell/pose-hg-train)
	
	Modification are made and explained in the report
	Goal : Achieve Real Time detection (Webcam)
	----- Modifications made to obtain faster results (trade off speed/accuracy)
	
	This work is free of use, please cite the author if you use it!
"""
import time
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import numpy as np
import sys
import datetime
import os
import matplotlib.pyplot as plt
import cv2
cv2.ocl.setUseOpenCL(False)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
'''
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
'''

class HourglassModel():
    """ HourglassModel class: (to be renamed)
	Generate TensorFlow model to train and predict Human Pose from images (soon videos)
	Please check README.txt for further information on model management.
	"""

    def __init__(self, nFeat=512, nStack=4, nModules=1, nLow=4, outputDim=16, batch_size=16, drop_rate=0.2,
                 lear_rate=2.5e-4, decay=0.96, decay_step=2000, dataset=None, training=True, w_summary=True,
                 logdir_train=None, logdir_test=None, tiny=True, attention=False, modif=True, w_loss=False,
                 name='tiny_hourglass',
                 joints=['r_anckle', 'r_knee', 'r_hip', 'l_hip', 'l_knee', 'l_anckle', 'pelvis', 'thorax', 'neck',
                         'head', 'r_wrist', 'r_elbow', 'r_shoulder', 'l_shoulder', 'l_elbow', 'l_wrist']):
        """ Initializer
		Args:
			nStack				: number of stacks (stage/Hourglass modules)
			nFeat				: number of feature channels on conv layers
			nLow				: number of downsampling (pooling) per module
			outputDim			: number of output Dimension (16 for MPII)
			batch_size			: size of training/testing Batch
			dro_rate			: Rate of neurons disabling for Dropout Layers
			lear_rate			: Learning Rate starting value
			decay				: Learning Rate Exponential Decay (decay in ]0,1], 1 for constant learning rate)
			decay_step			: Step to apply decay
			dataset			: Dataset (class DataGenerator)
			training			: (bool) True for training / False for prediction
			w_summary			: (bool) True/False for summary of weight (to visualize in Tensorboard)
			tiny				: (bool) Activate Tiny Hourglass
			attention			: (bool) Activate Multi Context Attention Mechanism (MCAM)
			modif				: (bool) Boolean to test some network modification # DO NOT USE IT ! USED TO TEST THE NETWORK
			name				: name of the model
		"""
        self.nStack = nStack
        self.nFeat = nFeat
        self.nModules = nModules
        self.outDim = outputDim
        self.batchSize = batch_size
        self.training = training
        self.w_summary = w_summary
        self.tiny = tiny
        self.dropout_rate = drop_rate
        self.learning_rate = lear_rate
        self.decay = decay
        self.name = name
        self.attention = attention
        self.decay_step = decay_step
        self.nLow = nLow
        self.modif = modif
        self.dataset = dataset
        self.cpu = '/cpu:0'
        self.gpu = '/gpu:0'   #'gpu:0
        self.logdir_train = logdir_train
        self.logdir_test = logdir_test
        self.joints = joints
        self.w_loss = w_loss
        self.point_pairs = [[0, 1], [1, 2], [2, 6], [3, 6], [4, 3], [5,4],[6,7],[10,11],[11,12],[12,7],[7,13],[13,14],[14,15],[8,7],[9,8],[15,14]]

        #新加的部分，为了进行跨层的信息传输
        self.stage_stack = 0
        self.One_stage = []
        self.One_stage.append(1)
        self.One_stage.append(2)
        self.One_stage.append(3)
        self.One_stage.append(4)
        self.Two_stage = []


    # ACCESSOR


    def get_input(self):
        """ Returns Input (Placeholder) Tensor
		Image Input :
			Shape: (None,256,256,3)
			Type : tf.float32
		Warning:
			Be sure to build the model first
		"""
        return self.img

    def get_output(self):
        """ Returns Output Tensor
		Output Tensor :
			Shape: (None, nbStacks, 64, 64, outputDim)
			Type : tf.float32
		Warning:
			Be sure to build the model first
		"""
        return self.output

    def get_label(self):
        """ Returns Label (Placeholder) Tensor
		Image Input :
			Shape: (None, nbStacks, 64, 64, outputDim)
			Type : tf.float32
		Warning:
			Be sure to build the model first
		"""
        return self.gtMaps

    def get_loss(self):
        """ Returns Loss Tensor
		Image Input :
			Shape: (1,)
			Type : tf.float32
		Warning:
			Be sure to build the model first
		"""
        return self.loss

    def get_saver(self):
        """ Returns Saver
		/!\ USE ONLY IF YOU KNOW WHAT YOU ARE DOING
		Warning:
			Be sure to build the model first
		"""
        return self.saver

    def generate_model(self):
        """ Create the complete graph
		"""
        startTime = time.time()
        print('CREATE MODEL:')
        with tf.device(self.gpu):
            with tf.name_scope('inputs'):
                # Shape Input Image - batchSize: None, height: 256, width: 256, channel: 3 (RGB)
                self.img = tf.placeholder(dtype=tf.float32, shape=(None, 256, 256, 3), name='input_img')
                ##不清楚这里是干什么的？
                if self.w_loss:
                    self.weights = tf.placeholder(dtype=tf.float32, shape=(None, self.outDim))

                # Shape Ground Truth Map: batchSize x nStack x 64 x 64 x outDim
                ##这个和堆叠沙漏网络最终的输出：是指这个么？，还是指在tf.stack(4个沙漏的输出结果组成的数组)
                self.gtMaps = tf.placeholder(dtype=tf.float32, shape=(None, self.nStack, 64, 64, self.outDim))

            # TODO : Implement weighted loss function
            # NOT USABLE AT THE MOMENT
            # weights = tf.placeholder(dtype = tf.float32, shape = (None, self.nStack, 1, 1, self.outDim))
            inputTime = time.time()
            print('---Inputs : Done (' + str(int(abs(inputTime - startTime))) + ' sec.)')

            ##以下这个是判断是否用'上下文注意力机制'，本实验原作者是没用的
            if self.attention:
                self.output = self._graph_mcam(self.img)
            else:
                self.output = self._graph_hourglass(self.img)

            graphTime = time.time()
            print('---Graph : Done (' + str(int(abs(graphTime - inputTime))) + ' sec.)')
            ##在tensorflow中图的建立是仅仅如此么，输入和输出的定义和神经网络的结构设计！

            with tf.name_scope('loss'):
                if self.w_loss:
                    self.loss = tf.reduce_mean(self.weighted_bce_loss(), name='reduced_loss')
                else:
                    #valid_mask = tf.reshape(weight, [self.batch_size, self.nStack, 1, 1, 16])
                    #print('valid_mask:',valid_mask)
                    #self.loss = tf.reduce_mean(
                    #    tf.reduce_mean(tf.square(self.output - self.gtMaps) * valid_mask)
                    #)
                    #原始代码中的损失为交叉熵损失，我将改为MSE损失
                    '''
                    all_loss = 0  # 所有阶段的损失之和（有nstack个阶段）
                    stage_loss = [0]*self.nStack
                    valid_mask = self.weights
                    for stage in range(self.nStack):
                        stage_loss_batch = [0]*8
                        for batch in range(8):
                            stage_loss_batch_hmindex = [0]*16
                            for hmindex in range(16):
                                stage_loss_batch_hmindex[hmindex] = tf.nn.l2_loss((self.gtMaps[batch,stage,:,:,hmindex])-(self.output[batch,stage,:,:,hmindex]))*valid_mask[batch][hmindex]
                            stage_loss_batch[batch] = tf.reduce_sum(stage_loss_batch_hmindex)
                        stage_loss[stage] = tf.reduce_sum(stage_loss_batch)/8

                    for stage in range(self.nStack):
                        all_loss += stage_loss[stage]
                    self.loss = all_loss
                    '''

                    '''
                    for stage in range(self.nStack):
                        heatmap_pred = self.output[:, stage, :, :, :]
                        heatmap_gt = self.gtMaps[:, stage, :, :, :]
                        self.loss = tf.reduce_mean((tf.square(heatmap_pred - heatmap_gt)))
                        all_loss += self.loss
                        self.loss = 0
                    self.loss = all_loss
                    '''

                    self.loss = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output, labels=self.gtMaps),
                        name='cross_entropy_loss')

            lossTime = time.time()
            print('---Loss : Done (' + str(int(abs(graphTime - lossTime))) + ' sec.)')

        ##准确度、学习率，训练的迭代次数
        with tf.device(self.cpu):
            with tf.name_scope('accuracy'):
                self._accuracy_computation()  ##返回一个一维数组（数组长度为关键点个数）这个准确率，
            # 是通过：一个batch图像数据都去求某个关键点的损失（加起来/图像总数）
            accurTime = time.time()
            print('---Acc : Done (' + str(int(abs(accurTime - lossTime))) + ' sec.)')

            with tf.name_scope('steps'):
                self.train_step = tf.Variable(0, name='global_step', trainable=False)

            with tf.name_scope('lr'):
                self.lr = tf.train.exponential_decay(self.learning_rate, self.train_step, self.decay_step, self.decay,
                                                     staircase=True, name='learning_rate')
            lrTime = time.time()
            print('---LR : Done (' + str(int(abs(accurTime - lrTime))) + ' sec.)')

        with tf.device(self.gpu):
            ##优化器、优化过程（降低loss）
            with tf.name_scope('rmsprop'):
                self.rmsprop = tf.train.RMSPropOptimizer(learning_rate=self.lr)
            optimTime = time.time()
            print('---Optim : Done (' + str(int(abs(optimTime - lrTime))) + ' sec.)')

            with tf.name_scope('minimizer'):
                self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(self.update_ops):
                    self.train_rmsprop = self.rmsprop.minimize(self.loss, self.train_step)
            minimTime = time.time()
            print('---Minimizer : Done (' + str(int(abs(optimTime - minimTime))) + ' sec.)')

        ##初始化所有的变量
        self.init = tf.global_variables_initializer()
        initTime = time.time()
        print('---Init : Done (' + str(int(abs(initTime - minimTime))) + ' sec.)')

        ##以下部分就是为了用tensorboard显示出需要的变量和训练过程的结果等
        ##训练过程中的  loss、lr、针对每个关键点的准确率（joint_accuracy 和 accuracy之间是什么关系？）
        with tf.device(self.cpu):
            with tf.name_scope('training'):
                tf.summary.scalar('loss', self.loss, collections=['train'])
                tf.summary.scalar('learning_rate', self.lr, collections=['train'])
            with tf.name_scope('summary'):
                for i in range(len(self.joints)):
                    tf.summary.scalar(self.joints[i], self.joint_accur[i], collections=['train', 'test'])

        self.train_op = tf.summary.merge_all('train')
        self.test_op = tf.summary.merge_all('test')
        self.weight_op = tf.summary.merge_all('weight')
        endTime = time.time()
        print('Model created (' + str(int(abs(endTime - startTime))) + ' sec.)')
        del endTime, startTime, initTime, optimTime, minimTime, lrTime, accurTime, lossTime, graphTime, inputTime

    def restore(self, load=None):
        """ Restore a pretrained model
		Args:
			load	: Model to load (None if training from scratch) (see README for further information)
		"""
        with tf.name_scope('Session'):
            with tf.device(self.gpu):
                self._init_session()
                self._define_saver_summary(summary=False)
                if load is not None:
                    print('Loading Trained Model')
                    t = time.time()
                    self.saver.restore(self.Session, load)
                    print('Model Loaded (', time.time() - t, ' sec.)')
                else:
                    print('Please give a Model in args (see README for further information)')

    def vis(self,img_cv2,points,points_g,point_pairs,epoch,heatmap_1,fun,stage,i):
        img_cv3 = img_cv2
        img_cv3 = np.clip(img_cv3 , 0, 255)
        img_cv2_copy = np.copy(img_cv3)


        '''
        points_new = []
        points_1 = points
        
        for i in range(16):
            x = (64 * points_1[i][0]) / 256
            y = (64 * points_1[i][1]) / 256
            points_new.append((int(x), int(y)))
        points_new = np.reshape(points_new, (-1, 2))

        #把关键点按照HR-net中的那种显示出来，是热力图的形式
        resized_image = cv2.resize(img_cv2_copy, (64, 64))

        #以下这个具体数字，我是直接改了的，
        grid_image = np.zeros(((1) * 64,
                               (16 + 1) * 64,
                               3),
                              dtype=np.uint8)
        for j in range(len(points_new)):
            cv2.circle(resized_image, tuple(points_new[j]),
                       1, [0, 0, 255], 1)
            heatmap = heatmap_1[:,:, j]  #这里我刚开始在函数的形式参数中就用heatmap就会报错，我觉得变量还是要改好
            #heatmap = np.clip(heatmap, 0, 255)
            heatmap = np.array(heatmap, np.uint8)
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image = colored_heatmap * 0.7 + resized_image * 0.3
            cv2.circle(masked_image,
                       tuple(points_new[j]),
                       1, [0, 0, 255], 1)
            #以下内容：因为我是直接就针对batch——size中的第一张进行，所以没有batch——size的高， HR-net中的vis部分的代码copy
            height_begin = 0
            height_end = 64
            width_begin = 64 * (j + 1)
            width_end = 64 * (j + 2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = masked_image
        grid_image[height_begin:height_end, 0:64, :] = resized_image
        
        plt.imshow(grid_image)
        plt.axis('off')
        plt.savefig('D:/home_program/xiangmu_1/data_test/VIS_ZJ/heat_map1/' + str(epoch))
       
        cv2.imwrite('D:/home_program/xiangmu_1/data_test/VIS_ZJ/heat_map1/' +str(epoch)+ '.jpg', grid_image)
        '''

        #添加循环的画在所有图上，我看看是否有符合的？

        for jdx in range(1):
            img_cv2_copy_1 = img_cv2_copy[jdx][:,:,:]
            img_cv2_copy_1 = img_cv2_copy_1.reshape(256, 256, 3)
            img_cv3_1 = np.copy(img_cv2_copy_1)
            for idx in range(len(points)):
                cv2.circle(img_cv2_copy_1,
                           tuple(points[idx]),
                           radius=3,
                           color=(0, 0, 255),
                           thickness=-1,
                           lineType=cv2.FILLED)
                cv2.circle(img_cv2_copy_1,
                           tuple(points_g[idx]),
                           radius=3,
                           color=(255, 0, 255),
                           thickness=-1,
                           lineType=cv2.FILLED)
                # Draw Skeleton
            for pair in point_pairs:
                partA = pair[0]
                partB = pair[1]
                if list(points[partA])[0] != -1 and list(points[partA])[1] != -1 and list(points[partB])[0] != -1 and \
                        list(points[partB])[1] != -1:
                    cv2.line(img_cv3_1,
                             tuple(points[partA]),
                             tuple(points[partB]),
                             color=(0, 255, 255), thickness=2)
                    cv2.circle(img_cv3_1,
                               tuple(points[partA]),
                               radius=3,
                               color=(0, 0, 255),
                               thickness=-1,
                               lineType=cv2.FILLED)

            plt.figure(figsize=[50, 50])
            plt.subplot(1, 2, 1)
            # plt.imshow(cv2.cvtColor(img_cv3, cv2.COLOR_BGR2RGB))
            plt.imshow(img_cv3_1)
            plt.axis("off")
            plt.subplot(1, 2, 2)
            plt.imshow(img_cv2_copy_1)
            # plt.imshow(cv2.cvtColor(img_cv2_copy, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.savefig('F:/zwx/STAGE3_NEW/Stage3[2,3]/0228/key_vis/' + str(epoch) + '_' + str(stage) + '_'+str(i) + str(fun))
            #plt.savefig('F:/wenxin_/Experiment_7.7/aug_7/key_vis/' + str(epoch) + '_'+ str(jdx) + str(fun))
            # plt.savefig('D:/home_program/xiangmu_1/data_test/VIS_ZJ/key_vis/'+ str(epoch))
            # plt.show()






    def get_point(self,demo_stage_heatmap,ground_heatmap,img_height,img_width,num_points,H,W,stage,epoch,i):
        points = []
        points_g = []
        for idx in range(num_points):
            probMap =demo_stage_heatmap[:,:,idx]
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
            x = (img_width * point[0]) / W
            y = (img_height * point[1]) / H
            if prob > 0.1: #0.1为阈值
                points.append((int(x), int(y)))
            else:
                points.append((-1,-1))

        for idx in range(num_points):
            probMap_1 =ground_heatmap[:,:,idx]
            minVal, prob, minLoc, point_ = cv2.minMaxLoc(probMap_1)
            x = (img_width * point_[0]) / W
            y = (img_height * point_[1]) / H
            if prob > 0.1: #0.1为阈值
                points_g.append((int(x), int(y)))
            else:
                points_g.append((-1,-1))

        logfile = 'F:/zwx/STAGE3_NEW/Stage3[2,3]/0225/joints.txt'
        fobj = open(logfile, 'a')
        fobj.write('epoch: ' + str(epoch) + '\n')
        fobj.write('stage: ' + str(stage) + '\n')
        fobj.write('joints: ' + str(points) + '\n')
        fobj.write('valide_iter: ' + str(i) + '\n')
        fobj.write('---------------------------------------------------------------------------\n')
        fobj.write('\n')
        fobj.close()

        return points,points_g


    def _train(self, nEpochs=100, epochSize=1000, saveStep=500, validIter=8): #这里默认为10次
        with tf.name_scope('Train'):
            ##生成一个batch的训练数据、验证数据

            self.generator = self.dataset._aux_generator(self.batchSize, self.nStack, normalize=True,
                                                         sample_set='train')
            self.valid_gen = self.dataset._aux_generator(self.batchSize, self.nStack, normalize=True,
                                                         sample_set='valid')

            startTime = time.time()
            self.resume = {}
            self.resume['accur'] = []  ##在训练了一个epoch后，在验证数据集上进行多次迭代验证，而这里存储的是最后一次的结果 accuracy_pred
            self.resume['loss'] = []  ##存储的每个epoch的cost，而这个cost是一个epoch中总共迭代时c的总和
            self.resume['err'] = []  ##将计算的每个关键点的准确率求平均值
            for epoch in range(nEpochs):
                epochstartTime = time.time()
                avg_cost = 0.
                cost = 0.
                print('Epoch :' + str(epoch) + '/' + str(nEpochs) + '\n')
                # Training Set

                accuracy_array_train = np.array([0.0] * len(self.joint_accur))  #新添加的内容

                for i in range(epochSize):
                    # DISPLAY PROGRESS BAR
                    # TODO : Customize Progress Bar
                    ##以下的进度条是怎么计算出来的
                    percent = ((i + 1) / epochSize) * 100
                    num = np.int(20 * percent / 100)
                    tToEpoch = int((time.time() - epochstartTime) * (100 - percent) / (percent))
                    sys.stdout.write(
                        '\r Train: {0}>'.format("=" * num) + "{0}>".format(" " * (20 - num)) + '||' + str(percent)[
                                                                                                      :4] + '%' + ' -cost: ' + str(
                            cost)[:6] + ' -avg_loss: ' + str(avg_cost)[:5] + ' -timeToEnd: ' + str(tToEpoch) + ' sec.')
                    sys.stdout.flush()

                    img_train, gt_train, weight_train = next(self.generator)
                    #1、我要确保关键点和图像是可以对应上的，




                    ##这里的saveStep是指：在训练过程中保存过程到log_train中
                    if i % saveStep == 0:
                        if self.w_loss:
                            _, c, summary = self.Session.run([self.train_rmsprop, self.loss, self.train_op],
                                                             feed_dict={self.img: img_train, self.gtMaps: gt_train,
                                                                        self.weights: weight_train})
                        else:
                            ##为什么在w_loss时有train_weight？？？
                            out_,joint_train_acur, _, c, summary = self.Session.run([self.output, self.joint_accur,self.train_rmsprop, self.loss, self.train_op],
                                                             feed_dict={self.img: img_train, self.gtMaps: gt_train})

                        # Save summary (Loss + Accuracy)
                        self.train_summary.add_summary(summary, epoch * epochSize + i)
                        self.train_summary.flush()
                    else:
                        if self.w_loss:
                            _, c, = self.Session.run([self.train_rmsprop, self.loss],
                                                     feed_dict={self.img: img_train, self.gtMaps: gt_train,
                                                                self.weights: weight_train})
                        else:
                            _, joint_train_acur,c, out_ = self.Session.run([self.train_rmsprop,self.joint_accur, self.loss,self.output],
                                                     feed_dict={self.img: img_train, self.gtMaps: gt_train})  #想在这里加个算准确率的那个

                    ## c为在一个epoch中一次迭代过程的误差，avg_cost为一个epoch下的所有迭代过程的误差相加除以总个数
                    ## 注意：这里在进入新的epoch后，avg_cost = 0.  cost = 0.
                    cost += c
                    avg_cost += c / epochSize

                    accuracy_array_train += np.array(joint_train_acur, dtype=np.float32) / epochSize

                #print('epoch: ',epoch)
                #print('accuracy_array_train:',accuracy_array_train)
                epochfinishTime = time.time()

                # Save Weight (axis = epoch)
                if self.w_loss:
                    weight_summary = self.Session.run(self.weight_op, {self.img: img_train, self.gtMaps: gt_train,
                                                                       self.weights: weight_train})
                else:
                    weight_summary = self.Session.run(self.weight_op, {self.img: img_train, self.gtMaps: gt_train})
                self.train_summary.add_summary(weight_summary, epoch)
                self.train_summary.flush()

                # self.weight_summary.add_summary(weight_summary, epoch)
                # self.weight_summary.flush()
                print('Epoch ' + str(epoch) + '/' + str(nEpochs) + ' done in ' + str(
                    int(epochfinishTime - epochstartTime)) + ' sec.' + ' -avg_time/batch: ' + str(
                    ((epochfinishTime - epochstartTime) / epochSize))[:4] + ' sec.')

                with tf.name_scope('save'):
                    ##这里的name也即在config.cfg中定义的'name'
                    #self.saver.save(self.Session, os.path.join(, str(self.name + '_' + str(epoch + 1))))
                    self.saver.save(self.Session, os.path.join(os.getcwd(),'0228/save_model', str(self.name + '_' + str(epoch + 1))))
                self.resume['loss'].append(cost)
                #1.save_model + save_model——7.12

                # Validation Set
                ##accuracy_array 是所有验证迭代次数的准确率平均值（在每次一个epoch训练后）
                accuracy_array = np.array([0.0] * len(self.joint_accur))

                #每隔多少epoch，将预测的热图的关键点预测出来并画在原图上
                #if epoch % 5 == 0 and epoch >= int(nEpochs/2):
                if epoch == 100:
                    fun = 'train_data'
                    #if epoch % 2 == 0 and epoch >= nEpochs/3:  #在规定的epoch的1/3就开始显示这个
                    bs, stack, size_height, size_width, weidu = out_.shape
                    for stage in range(stack):
                        #demo_stage_heatmap = out_[0,stage,:, :, 0:weidu]
                        demo_stage_heatmap = out_[0, stage,:, :, 0:weidu]
                        demo_stage_heatmap = demo_stage_heatmap.reshape(64, 64, 16)
                        demo_stage_heatmap = cv2.resize(demo_stage_heatmap, (256, 256))
                        plt.imshow(img_train[0])
                        plt.xlabel('train_image')
                        plt.figure(figsize=[50, 50])
                        for i in range(16):
                            plt.subplot(4, 4, i + 1)
                            plt.imshow(img_train[0])
                            # demo_stage_heatmap[:,:,i] = cv2.applyColorMap(demo_stage_heatmap[:,:,i], cv2.COLORMAP_JET)
                            plt.imshow(demo_stage_heatmap[:, :, i], alpha=0.6)
                            plt.colorbar()
                            plt.axis('off')
                        plt.savefig('F:/zwx/STAGE3_NEW/Stage3[2,3]/0225/m_results/' + str(epoch) + '_' + str(stage))


                loss_valid = 0
                for i in range(validIter):
                    img_valid, gt_valid, w_valid = next(self.generator)
                    #plt.imshow(img_valid[0])
                    #plt.show()
                    ##一个batchsize图像数据的1-L2误差的平均值（因为是一个batch的数据）
                    ##这里也没有放入valid_weight
                    accuracy_pred,cost_valid,out_ = self.Session.run([self.joint_accur,self.loss,self.output],
                                                            feed_dict={self.img: img_valid, self.gtMaps: gt_valid})
                    ##保存中间结果：

                    batch_size, stack, size_height, size_width, weidu = out_.shape
                    batch_size_1, stack_1, size_height_1, size_width_1, weidu_1 = gt_valid.shape
                    if epoch == 100 :
                    #if epoch % 5 == 0 and epoch >= int(nEpochs / 8):  #原来这里是/8  /7
                        for stage in range(stack):
                            demo_stage_heatmap = out_[0,stage, :, :, 0:weidu]
                            #demo_stage_heatmap = out_[stage][0, :, :, 0:weidu]
                            # print('demo_stage_heatmap.shape:',demo_stage_heatmap.shape)
                            demo_stage_heatmap = demo_stage_heatmap.reshape(64, 64, 16)
                            #print('gt_valid:',gt_valid.shape) (8, 3, 64, 64, 16)
                            gt_valid_1 = gt_valid[0,stage, :, :, 0:weidu_1]
                            gt_valid_1 = gt_valid_1.reshape(64,64,16)
                            if stage == 2 and i % 4 == 0:
                                fun = 'valid_data'
                                h, w, _ = img_train[0].shape
                                # 找到每个热图对应的最大响应的关键点坐标
                                # 为使得这里能够进行下去，因为之前在points很可能得到包含'None'这类元素，就无法继续执行代码
                                points,points_g = self.get_point(demo_stage_heatmap, gt_valid_1,h, w, weidu, size_height, size_width,stage,epoch,i)
                                points = np.reshape(points, (-1, 2))
                                points_g = np.reshape(points_g,(-1,2))
                                # print('points:',points)
                                # self.vis(img_train[0],points,self.point_pairs,epoch,demo_stage_heatmap)
                                self.vis(img_valid, points,points_g, self.point_pairs, epoch, demo_stage_heatmap,fun,stage,i)

                        '''
                                demo_stage_heatmap = cv2.resize(demo_stage_heatmap, (256, 256))
                            # demo_stage_heatmap = demo_stage_heatmap.reshape(self.hm_size,self.hm_size,self.num_joints)
                            # demo_stage_heatmap = cv2.resize(demo_stage_heatmap,(self.image_size, self.image_size))
                            # 最终的结果：demo_stage_heatmap的形状：(256,256,16)
                            #if stage == 3 and i % 4 == 0:  # 只保存其中两次即可
                                plt.figure(figsize=[25, 25])
                                for j in range(16):
                                    plt.subplot(4, 4, j + 1)
                                    #lt.imshow(img_valid[0])
                                    heatmap = demo_stage_heatmap[:, :, j]
                                    #heatmap = np.clip(heatmap, 0, 255)
                                    heatmap = np.array(heatmap, np.uint8)
                                    colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                                    masked_image = colored_heatmap * 0.6 + img_valid[0] * 0.4
                                    #cv2.circle(masked_image,
                                    #            tuple(points[j]),
                                     #          1, [0, 0, 255], 1)
                                    # print('heatmap.shape:',heatmap.shape)
                                    # heatmap = np.reshape(heatmap, (256, 256, 1))
                                    plt.imshow(masked_image)
                                    #plt.imshow(colored_heatmap, alpha=0.6)
                                    plt.colorbar()
                                    plt.axis('off')
                                plt.savefig('E:/results/heat_map/' + str(stage) + '_' + str(i) + '_' + str(epoch))
                                # plt.savefig('D:/home_program/xiangmu_1/data_test/VIS_ZJ/heat_map/' + str(
                                #    stage) + '_' + str(i) + '_' + str(epoch))
                                # plt.show()
                    '''

                    loss_valid = loss_valid + cost_valid  #这样的loss肯定小啊，因为这里面的验证图像个数是少的，
                    accuracy_array += np.array(accuracy_pred, dtype=np.float32) / validIter
                    #print('i:',i)
                    #print('--accuracy_pred = ',accuracy_pred)  #我猜测这个是一个列表，共16维度（每个维度代表某个关键点）
                #print('loss_valid: ',loss_valid)
                #print('--aaccuracy_array = ',accuracy_array)  #最终就比较这个就行！（train和valid）
                print('--Avg. Accuracy =', str((np.sum(accuracy_array) / len(accuracy_array)) * 100)[:6], '%') #在验证集上的准确率


                #记录下中间结果：
                logfile = 'F:/zwx/STAGE3_NEW/Stage3[2,3]/0228/record.txt'
                fobj = open(logfile, 'a')
                fobj.write('epoch: ' + str(epoch) + '\n')
                fobj.write('accuracy_array_train: ' + str(accuracy_array_train) + '\n')
                fobj.write('--aaccuracy_array: ' + str(accuracy_array) + '\n')
                fobj.write('----Avg. Accuracy : ' + str((np.sum(accuracy_array) / len(accuracy_array)) * 100)[:6])
                fobj.write('\n')
                fobj.write('cost_train: ' + str(cost) + '\n')
                fobj.write('---------------------------------------------------------------------------\n')
                fobj.write('\n')
                fobj.close()

                self.resume['accur'].append(accuracy_pred)
                ##误差这么求么？
                self.resume['err'].append(np.sum(accuracy_array) / len(accuracy_array))

                valid_summary = self.Session.run(self.test_op, feed_dict={self.img: img_valid, self.gtMaps: gt_valid})
                self.test_summary.add_summary(valid_summary, epoch)
                self.test_summary.flush()

            print('Training Done')
            print('Resume:' + '\n' + '  Epochs: ' + str(nEpochs) + '\n' + '  n. Images: ' + str(
                nEpochs * epochSize * self.batchSize))
            print('  Final Loss: ' + str(cost) + '\n' + '  Relative Loss: ' + str(
                100 * self.resume['loss'][-1] / (self.resume['loss'][0] + 0.1)) + '%')
            print('  Relative Improvement: ' + str((self.resume['err'][-1] - self.resume['err'][0]) * 100) + '%')
            print('  Training Time: ' + str(datetime.timedelta(seconds=time.time() - startTime)))

    def record_training(self, record):
        """ Record Training Data and Export them in CSV file
		Args:
			record		: record dictionnary
		"""
        out_file = open(self.name + '_train_record.csv', 'w')
        for line in range(len(record['accur'])):
            out_string = ''
            labels = [record['loss'][line]] + [record['err'][line]] + record['accur'][line]
            for label in labels:
                out_string += str(label) + ', '
            out_string += '\n'
            out_file.write(out_string)
        out_file.close()
        print('Training Record Saved')

    ##默认定义 def training_init(self, nEpochs = 10, epochSize = 1000, saveStep = 500, dataset = None, load = None):
    def training_init(self,nEpochs=100, epochSize=2500, saveStep=100, dataset=None, load = None):
        #load = 'E:/xiao/Data_new_7.16/STAGE3_NEW/Stage3[2,3]/Save3_conect2and3/hg_network_55'
        # load='E:/results/model_3/hg_network_152')
        ##并未将数据集传进来，但其实里面涉及了数据集的操作
        """ Initialize the training
		Args:
			nEpochs		: Number of Epochs to train
			epochSize		: Size of one Epoch
			saveStep		: Step to save 'train' summary (has to be lower than epochSize)
			dataset		: Data Generator (see generator.py)
			load			: Model to load (None if training from scratch) (see README for further information)
		"""
        with tf.name_scope('Session'):
            with tf.device(self.gpu):
                ##以下初始化就是在generate_model函数的 init
                self._init_weight()
                self._define_saver_summary()

                ##这里注意下：我个人认为和预训练模型有关
                ##不清楚这个是如何用的？——有关预训练模型
                if load is not None:
                    self.saver.restore(self.Session, load)
                # try:
                #	self.saver.restore(self.Session, load)
                # except Exception:
                #	print('Loading Failed! (Check README file for further information)')
                self._train(nEpochs, epochSize, saveStep, validIter = 5)


    def weighted_bce_loss(self):
        """ Create Weighted Loss Function
		WORK IN PROGRESS
		"""
        self.bceloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output, labels=self.gtMaps),
                                      name='cross_entropy_loss')
        e1 = tf.expand_dims(self.weights, axis=1, name='expdim01')
        e2 = tf.expand_dims(e1, axis=1, name='expdim02')
        e3 = tf.expand_dims(e2, axis=1, name='expdim03')
        return tf.multiply(e3, self.bceloss, name='lossW')

    def _accuracy_computation(self):
        """ Computes accuracy tensor
		"""
        self.joint_accur = []
        for i in range(len(self.joints)):
            self.joint_accur.append(
                self._accur(self.output[:, self.nStack - 1, :, :, i], self.gtMaps[:, self.nStack - 1, :, :, i],
                            self.batchSize))

    def _define_saver_summary(self, summary=True):
        """ Create Summary and Saver
		Args:
			logdir_train		: Path to train summary directory
			logdir_test		: Path to test summary directory
		"""
        if (self.logdir_train == None) or (self.logdir_test == None):
            raise ValueError('Train/Test directory not assigned')
        else:
            with tf.device(self.cpu):
                self.saver = tf.train.Saver(max_to_keep=100)
            if summary:
                with tf.device(self.gpu):
                    self.train_summary = tf.summary.FileWriter(self.logdir_train, tf.get_default_graph())
                    self.test_summary = tf.summary.FileWriter(self.logdir_test)
                # self.weight_summary = tf.summary.FileWriter(self.logdir_train, tf.get_default_graph())

    def _init_weight(self):
        """ Initialize weights
		"""
        print('Session initialization')

        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33)
        # self.Session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        #设置最小的gpu使用量
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        #config = tf.ConfigProto()
        #config.gpu_options.per_process_gpu_memory_fraction = 0.6
        #config.allow_soft_placement = True
        self.Session = tf.Session(config=config)


        #self.Session = tf.Session()
        t_start = time.time()
        self.Session.run(self.init)
        print('Sess initialized in ' + str(int(time.time() - t_start)) + ' sec.')

    def _init_session(self):
        """ Initialize Session
		"""
        print('Session initialization')
        t_start = time.time()
        self.Session = tf.Session()
        '''
        if 'Session' in locals() and self.Session is not None:
            print('Close interactive session')
            self.Session.close()
        '''
        print('Sess initialized in ' + str(int(time.time() - t_start)) + ' sec.')

    def _graph_hourglass(self, inputs):
        """Create the Network
		Args:
			inputs : TF Tensor (placeholder) of shape (None, 256, 256, 3) #TODO : Create a parameter for customize size
		"""
        with tf.name_scope('model'):
            with tf.name_scope('preprocessing'):
                # Input Dim : nbImages x 256 x 256 x 3
                pad1 = tf.pad(inputs, [[0, 0], [2, 2], [2, 2], [0, 0]], name='pad_1')
                # Dim pad1 : nbImages x 260 x 260 x 3
                conv1 = self._conv_bn_relu(pad1, filters=64, kernel_size=6, strides=2, name='conv_256_to_128')
                # Dim conv1 : nbImages x 128 x 128 x 64
                r1 = self._residual(conv1, numOut=128, name='r1')
                # Dim pad1 : nbImages x 128 x 128 x 128
                pool1 = tf.contrib.layers.max_pool2d(r1, [2, 2], [2, 2], padding='VALID')
                # Dim pool1 : nbImages x 64 x 64 x 128
                if self.tiny:
                    r3 = self._residual(pool1, numOut=self.nFeat, name='r3')
                else:
                    r2 = self._residual(pool1, numOut=int(self.nFeat / 2), name='r2')
                    r3 = self._residual(r2, numOut=self.nFeat, name='r3')

            ##以下这个就是一个初始化过程
            # Storage Table
            hg = [None] * self.nStack
            ll = [None] * self.nStack
            ll_ = [None] * self.nStack
            drop = [None] * self.nStack
            out = [None] * self.nStack
            out_ = [None] * self.nStack
            sum_ = [None] * self.nStack

            ##以下我猜测是个轻量级的沙漏网络
            if self.tiny:
                with tf.name_scope('stacks'):

                    with tf.name_scope('stage_0'):
                        hg[0] = self._hourglass(r3, self.nLow, self.nFeat, 'hourglass')
                        drop[0] = tf.layers.dropout(hg[0], rate=self.dropout_rate, training=self.training,
                                                    name='dropout')
                        ll[0] = self._conv_bn_relu(drop[0], self.nFeat, 1, 1, name='ll')
                        if self.modif:
                            # TEST OF BATCH RELU
                            out[0] = self._conv_bn_relu(ll[0], self.outDim, 1, 1, 'VALID', 'out')
                        else:
                            out[0] = self._conv(ll[0], self.outDim, 1, 1, 'VALID', 'out')
                        out_[0] = self._conv(out[0], self.nFeat, 1, 1, 'VALID', 'out_')
                        sum_[0] = tf.add_n([out_[0], ll[0], r3], name='merge')


                    for i in range(1, self.nStack - 1):
                        with tf.name_scope('stage_' + str(i)):
                            hg[i] = self._hourglass_2_(sum_[i - 1], self.nLow, self.nFeat, 'hourglass')
                            drop[i] = tf.layers.dropout(hg[i], rate=self.dropout_rate, training=self.training,
                                                        name='dropout')
                            ll[i] = self._conv_bn_relu(drop[i], self.nFeat, 1, 1, name='ll')
                            if self.modif:
                                # TEST OF BATCH RELU
                                out[i] = self._conv_bn_relu(ll[i], self.outDim, 1, 1, 'VALID', 'out')
                            else:
                                out[i] = self._conv(ll[i], self.outDim, 1, 1, 'VALID', 'out')
                            out_[i] = self._conv(out[i], self.nFeat, 1, 1, 'VALID', 'out_')
                            sum_[i] = tf.add_n([out_[i], ll[i], sum_[i - 1]], name='merge')

                    with tf.name_scope('stage_' + str(self.nStack - 1)):
                        hg[self.nStack - 1] = self._hourglass(sum_[self.nStack - 2], self.nLow, self.nFeat, 'hourglass')
                        drop[self.nStack - 1] = tf.layers.dropout(hg[self.nStack - 1], rate=self.dropout_rate,
                                                                  training=self.training, name='dropout')
                        ll[self.nStack - 1] = self._conv_bn_relu(drop[self.nStack - 1], self.nFeat, 1, 1, 'VALID',
                                                                 'conv')
                        if self.modif:
                            out[self.nStack - 1] = self._conv_bn_relu(ll[self.nStack - 1], self.outDim, 1, 1, 'VALID',
                                                                      'out')
                        else:
                            out[self.nStack - 1] = self._conv(ll[self.nStack - 1], self.outDim, 1, 1, 'VALID', 'out')
                if self.modif:
                    return tf.nn.sigmoid(tf.stack(out, axis=1, name='stack_output'), name='final_output')
                else:
                    return tf.stack(out, axis=1, name='final_output')

            ##而此项目采用的是以下的堆叠沙漏过程
            else:
                print("------------------------开始检查------------------------")
                with tf.name_scope('stacks'):
                    # 堆叠沙漏网络的第一个阶段
                    with tf.name_scope('stage_0'):
                        hg[0] = self._hourglass_3_(r3, self.nLow, self.nFeat, 'hourglass')
                        drop[0] = tf.layers.dropout(hg[0], rate=self.dropout_rate, training=self.training,
                                                    name='dropout')
                        ll[0] = self._conv_bn_relu(drop[0], self.nFeat, 1, 1, 'VALID', name='conv')
                        ll_[0] = self._conv(ll[0], self.nFeat, 1, 1, 'VALID', 'll')
                        if self.modif:
                            # TEST OF BATCH RELU
                            out[0] = self._conv_bn_relu(ll[0], self.outDim, 1, 1, 'VALID', 'out')
                        else:
                            out[0] = self._conv(ll[0], self.outDim, 1, 1, 'VALID', 'out')
                        out_[0] = self._conv(out[0], self.nFeat, 1, 1, 'VALID', 'out_')
                        sum_[0] = tf.add_n([out_[0], r3, ll_[0]],name='merge')

                    # 堆叠沙漏网络的第二阶段range（1，2）其实就是1
                    for i in range(1, self.nStack - 1):
                        with tf.name_scope('stage_' + str(i)):
                            hg[i] = self._hourglass(sum_[i - 1], self.nLow, self.nFeat, 'hourglass')
                            drop[i] = tf.layers.dropout(hg[i], rate=self.dropout_rate, training=self.training,
                                                        name='dropout')
                            ll[i] = self._conv_bn_relu(drop[i], self.nFeat, 1, 1, 'VALID', name='conv')
                            ll_[i] = self._conv(ll[i], self.nFeat, 1, 1, 'VALID', 'll')
                            if self.modif:
                                out[i] = self._conv_bn_relu(ll[i], self.outDim, 1, 1, 'VALID', 'out')
                            else:
                                out[i] = self._conv(ll[i], self.outDim, 1, 1, 'VALID', 'out')
                            out_[i] = self._conv(out[i], self.nFeat, 1, 1, 'VALID', 'out_')
                            sum_[i] = tf.add_n([out_[i], sum_[i - 1], ll_[0]], name='merge')

                    ##堆叠沙漏网络的最后一个阶段
                    with tf.name_scope('stage_' + str(self.nStack - 1)):
                        hg[self.nStack - 1] = self._hourglass_2_(sum_[self.nStack - 2], self.nLow, self.nFeat, 'hourglass')
                        drop[self.nStack - 1] = tf.layers.dropout(hg[self.nStack - 1], rate=self.dropout_rate,
                                                                  training=self.training, name='dropout')
                        ll[self.nStack - 1] = self._conv_bn_relu(drop[self.nStack - 1], self.nFeat, 1, 1, 'VALID',
                                                                 'conv')
                        if self.modif:
                            out[self.nStack - 1] = self._conv_bn_relu(ll[self.nStack - 1], self.outDim, 1, 1, 'VALID',
                                                                      'out')
                        else:
                            out[self.nStack - 1] = self._conv(ll[self.nStack - 1], self.outDim, 1, 1, 'VALID', 'out')

                if self.modif:
                    return tf.nn.sigmoid(tf.stack(out, axis=1, name='stack_output'), name='final_output')
                else:
                    ##把每次的过程结果堆叠在一起（每次过程：就是4次stack）
                    return tf.stack(out, axis=1, name='final_output')

    def _conv(self, inputs, filters, kernel_size=1, strides=1, pad='VALID', name='conv'):
        """ Spatial Convolution (CONV2D)
		Args:
			inputs			: Input Tensor (Data Type : NHWC)
			filters		: Number of filters (channels)
			kernel_size	: Size of kernel
			strides		: Stride
			pad				: Padding Type (VALID/SAME) # DO NOT USE 'SAME' NETWORK BUILT FOR VALID
			name			: Name of the block
		Returns:
			conv			: Output Tensor (Convolved Input)
		"""
        with tf.name_scope(name):
            # Kernel for convolution, Xavier Initialisation
            kernel = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)(
                [kernel_size, kernel_size, inputs.get_shape().as_list()[3], filters]), name='weights')
            conv = tf.nn.conv2d(inputs, kernel, [1, strides, strides, 1], padding=pad, data_format='NHWC')
            if self.w_summary:
                with tf.device('/cpu:0'):
                    tf.summary.histogram('weights_summary', kernel, collections=['weight'])
            return conv

    def _conv_bn_relu(self, inputs, filters, kernel_size=1, strides=1, pad='VALID', name='conv_bn_relu'):
        """ Spatial Convolution (CONV2D) + BatchNormalization + ReLU Activation
		Args:
			inputs			: Input Tensor (Data Type : NHWC)
			filters		: Number of filters (channels)
			kernel_size	: Size of kernel
			strides		: Stride
			pad				: Padding Type (VALID/SAME) # DO NOT USE 'SAME' NETWORK BUILT FOR VALID
			name			: Name of the block
		Returns:
			norm			: Output Tensor
		"""
        with tf.name_scope(name):
            kernel = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)(
                [kernel_size, kernel_size, inputs.get_shape().as_list()[3], filters]), name='weights')
            conv = tf.nn.conv2d(inputs, kernel, [1, strides, strides, 1], padding='VALID', data_format='NHWC')
            norm = tf.contrib.layers.batch_norm(conv, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu,
                                                is_training=self.training)
            if self.w_summary:
                with tf.device('/cpu:0'):
                    tf.summary.histogram('weights_summary', kernel, collections=['weight'])
            return norm

    def _conv_block(self, inputs, numOut, name='conv_block'):
        """ Convolutional Block
		Args:
			inputs	: Input Tensor
			numOut	: Desired output number of channel
			name	: Name of the block
		Returns:
			conv_3	: Output Tensor
		"""

        ##本作者的实现并未用tiny_hourglass
        if self.tiny:
            with tf.name_scope(name):
                norm = tf.contrib.layers.batch_norm(inputs, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu,
                                                    is_training=self.training)
                pad = tf.pad(norm, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]), name='pad')
                conv = self._conv(pad, int(numOut), kernel_size=3, strides=1, pad='VALID', name='conv')
                return conv
        else:
            with tf.name_scope(name):
                with tf.name_scope('norm_1'):
                    norm_1 = tf.contrib.layers.batch_norm(inputs, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu,
                                                          is_training=self.training)
                    conv_1 = self._conv(norm_1, int(numOut / 2), kernel_size=1, strides=1, pad='VALID', name='conv')
                with tf.name_scope('norm_2'):
                    norm_2 = tf.contrib.layers.batch_norm(conv_1, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu,
                                                          is_training=self.training)
                    pad = tf.pad(norm_2, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]), name='pad')
                    conv_2 = self._conv(pad, int(numOut / 2), kernel_size=3, strides=1, pad='VALID', name='conv')
                with tf.name_scope('norm_3'):
                    norm_3 = tf.contrib.layers.batch_norm(conv_2, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu,
                                                          is_training=self.training)
                    conv_3 = self._conv(norm_3, int(numOut), kernel_size=1, strides=1, pad='VALID', name='conv')
                return conv_3

    def _skip_layer(self, inputs, numOut, name='skip_layer'):
        """ Skip Layer
		Args:
			inputs	: Input Tensor
			numOut	: Desired output number of channel
			name	: Name of the bloc
		Returns:
			Tensor of shape (None, inputs.height, inputs.width, numOut)
		"""
        with tf.name_scope(name):
            if inputs.get_shape().as_list()[3] == numOut:
                return inputs
            else:
                conv = self._conv(inputs, numOut, kernel_size=1, strides=1, name='conv')
                return conv

    def _residual(self, inputs, numOut, name='residual_block'):
        """ Residual Unit
		Args:
			inputs	: Input Tensor
			numOut	: Number of Output Features (channels)
			name	: Name of the block
		"""
        with tf.name_scope(name):
            convb = self._conv_block(inputs, numOut)
            skipl = self._skip_layer(inputs, numOut)
            if self.modif:
                return tf.nn.relu(tf.add_n([convb, skipl], name='res_block'))
            else:
                return tf.add_n([convb, skipl], name='res_block')


    def _hourglass_4(self, inputs, n, numOut, name='hourglass_4'):
        """ Hourglass Module
    	Args:
    		inputs	: Input Tensor
    		n		: Number of downsampling step
    		numOut	: Number of Output Features (channels)
    		name	: Name of the block
    	"""
        with tf.name_scope(name):
            # Upper Branch
            up_1 = self._residual(inputs, numOut, name='up_1')
            # Lower Branch
            low_ = tf.contrib.layers.max_pool2d(inputs, [2, 2], [2, 2], padding='VALID')
            low_1 = self._residual(low_, numOut, name='low_1')

            if n > 0:
                low_2 = self._hourglass(low_1, n - 1, numOut, name='low_2')
            else:
                low_2 = self._residual(low_1, numOut, name='low_2')

            low_3 = self._residual(low_2, numOut, name='low_3')
            up_2 = tf.image.resize_nearest_neighbor(low_3, tf.shape(low_3)[1:3] * 2, name='upsampling')
            if self.modif:
                # Use of RELU
                return tf.nn.relu(tf.add_n([up_2, up_1]), name='out_hg')
            else:
                return tf.add_n([up_2, up_1], name='out_hg')


    def _hourglass_3(self, inputs, n, numOut, name='hourglass_3'):
        """ Hourglass Module
    	Args:
    		inputs	: Input Tensor
    		n		: Number of downsampling step
    		numOut	: Number of Output Features (channels)
    		name	: Name of the block
    	"""
        with tf.name_scope(name):
            # Upper Branch
            up_1 = self._residual(inputs, numOut, name='up_1')
            # Lower Branch
            low_ = tf.contrib.layers.max_pool2d(inputs, [2, 2], [2, 2], padding='VALID')
            low_1 = self._residual(low_, numOut, name='low_1')

            if n > 0:
                low_2 = self._hourglass_4(low_1, n - 1, numOut, name='low_2')
            else:
                low_2 = self._residual(low_1, numOut, name='low_2')

            low_3 = self._residual(low_2, numOut, name='low_3')
            up_2 = tf.image.resize_nearest_neighbor(low_3, tf.shape(low_3)[1:3] * 2, name='upsampling')
            if self.modif:
                # Use of RELU
                return tf.nn.relu(tf.add_n([up_2, up_1]), name='out_hg')
            else:
                self.One_stage[3] = tf.add_n([up_2, up_1])
                print('One_stage[3]:',self.One_stage[3],type(self.One_stage[3]))
                return tf.add_n([up_2, up_1], name='out_hg')


    def _hourglass_2(self, inputs, n, numOut, name='hourglass_2'):
        """ Hourglass Module
    	Args:
    		inputs	: Input Tensor
    		n		: Number of downsampling step
    		numOut	: Number of Output Features (channels)
    		name	: Name of the block
    	"""
        with tf.name_scope(name):
            # Upper Branch
            up_1 = self._residual(inputs, numOut, name='up_1')
            # Lower Branch
            low_ = tf.contrib.layers.max_pool2d(inputs, [2, 2], [2, 2], padding='VALID')
            low_1 = self._residual(low_, numOut, name='low_1')

            if n > 0:
                low_2 = self._hourglass_3(low_1, n - 1, numOut, name='low_2')
            else:
                low_2 = self._residual(low_1, numOut, name='low_2')

            low_3 = self._residual(low_2, numOut, name='low_3')
            up_2 = tf.image.resize_nearest_neighbor(low_3, tf.shape(low_3)[1:3] * 2, name='upsampling')
            if self.modif:
                # Use of RELU
                return tf.nn.relu(tf.add_n([up_2, up_1]), name='out_hg')
            else:
                self.One_stage[2] = tf.add_n([up_2, up_1])
                return tf.add_n([up_2, up_1], name='out_hg')

    def _hourglass_1(self, inputs, n, numOut, name='hourglass_1'):
        """ Hourglass Module
    	Args:
    		inputs	: Input Tensor
    		n		: Number of downsampling step
    		numOut	: Number of Output Features (channels)
    		name	: Name of the block
    	"""
        with tf.name_scope(name):
            # Upper Branch
            up_1 = self._residual(inputs, numOut, name='up_1')
            # Lower Branch
            low_ = tf.contrib.layers.max_pool2d(inputs, [2, 2], [2, 2], padding='VALID')
            low_1 = self._residual(low_, numOut, name='low_1')

            if n > 0:
                low_2 = self._hourglass_2(low_1, n - 1, numOut, name='low_2')
            else:
                low_2 = self._residual(low_1, numOut, name='low_2')

            low_3 = self._residual(low_2, numOut, name='low_3')
            up_2 = tf.image.resize_nearest_neighbor(low_3, tf.shape(low_3)[1:3] * 2, name='upsampling')
            if self.modif:
                # Use of RELU
                return tf.nn.relu(tf.add_n([up_2, up_1]), name='out_hg')
            else:
                self.One_stage[1] = tf.add_n([up_2, up_1])
                print('One_stage[1]:',self.One_stage[1])

                return tf.add_n([up_2, up_1], name='out_hg')

    def _hourglass(self, inputs, n, numOut, name='hourglass_all'):
        """ Hourglass Module
		Args:
			inputs	: Input Tensor
			n		: Number of downsampling step
			numOut	: Number of Output Features (channels)
			name	: Name of the block
		"""
        with tf.name_scope(name):
            # Upper Branch
            up_1 = self._residual(inputs, numOut, name='up_1')
            # Lower Branch
            low_ = tf.contrib.layers.max_pool2d(inputs, [2, 2], [2, 2], padding='VALID')
            low_1 = self._residual(low_, numOut, name='low_1')

            if n > 0:
                low_2 = self._hourglass_1(low_1, n - 1, numOut, name='low_2')
            else:
                low_2 = self._residual(low_1, numOut, name='low_2')

            low_3 = self._residual(low_2, numOut, name='low_3')
            up_2 = tf.image.resize_nearest_neighbor(low_3, tf.shape(low_3)[1:3] * 2, name='upsampling')
            if self.modif:
                # Use of RELU
                return tf.nn.relu(tf.add_n([up_2, up_1]))
            else:
                self.One_stage[0] = tf.add_n([up_2, up_1])
                print('One_stage[0]:', self.One_stage[0])
                return tf.add_n([up_2, up_1], name='out_hg')

    def _hourglass_3_4(self, inputs, n, numOut, name='hourglass_all'):
        """ Hourglass Module
    	Args:
    		inputs	: Input Tensor
    		n		: Number of downsampling step
    		numOut	: Number of Output Features (channels)
    		name	: Name of the block
    	"""
        with tf.name_scope(name):
            # Upper Branch
            up_1 = self._residual(inputs, numOut, name='up_1')
            # Lower Branch
            low_ = tf.contrib.layers.max_pool2d(inputs, [2, 2], [2, 2], padding='VALID')
            low_1 = self._residual(low_, numOut, name='low_1')

            if n > 0:
                low_2 = self._hourglass_3_(low_1, n - 1, numOut, name='low_2')
            else:
                low_2 = self._residual(low_1, numOut, name='low_2')

            low_3 = self._residual(low_2, numOut, name='low_3')
            up_2 = tf.image.resize_nearest_neighbor(low_3, tf.shape(low_3)[1:3] * 2, name='upsampling')
            if self.modif:
                # Use of RELU
                return tf.nn.relu(tf.add_n([up_2, up_1]))
            else:
                return tf.add_n([up_2, up_1], name='out_hg')

    def _hourglass_3_3(self, inputs, n, numOut, name='hourglass_all'):
        """ Hourglass Module
    	Args:
    		inputs	: Input Tensor
    		n		: Number of downsampling step
    		numOut	: Number of Output Features (channels)
    		name	: Name of the block
    	"""
        with tf.name_scope(name):
            # Upper Branch
            up_1 = self._residual(inputs, numOut, name='up_1')
            # Lower Branch
            low_ = tf.contrib.layers.max_pool2d(inputs, [2, 2], [2, 2], padding='VALID')
            low_1 = self._residual(low_, numOut, name='low_1')

            if n > 0:
                low_2 = self._hourglass_3_4(low_1, n - 1, numOut, name='low_2')
            else:
                low_2 = self._residual(low_1, numOut, name='low_2')

            low_3 = self._residual(low_2, numOut, name='low_3')
            up_2 = tf.image.resize_nearest_neighbor(low_3, tf.shape(low_3)[1:3] * 2, name='upsampling')
            if self.modif:
                # Use of RELU
                return tf.nn.relu(tf.add_n([up_2, up_1]))
            else:
                return tf.add_n([up_2, up_1], name='out_hg')

    def _hourglass_3_2(self, inputs, n, numOut, name='hourglass_all'):
        """ Hourglass Module
        Args:
            inputs	: Input Tensor
            n		: Number of downsampling step
            numOut	: Number of Output Features (channels)
            name	: Name of the block
        """
        with tf.name_scope(name):
            # Upper Branch
            up_1 = self._residual(inputs, numOut, name='up_1')
            # Lower Branch
            low_ = tf.contrib.layers.max_pool2d(inputs, [2, 2], [2, 2], padding='VALID')
            low_1 = self._residual(low_, numOut, name='low_1')

            if n > 0:
                low_2 = self._hourglass_3_3(low_1, n - 1, numOut, name='low_2')
            else:
                low_2 = self._residual(low_1, numOut, name='low_2')

            low_3 = self._residual(low_2, numOut, name='low_3')
            up_2 = tf.image.resize_nearest_neighbor(low_3, tf.shape(low_3)[1:3] * 2, name='upsampling')
            if self.modif:
                # Use of RELU
                return tf.nn.relu(tf.add_n([up_2, up_1]))
            else:
                return tf.add_n([up_2, up_1], name='out_hg')

    def _hourglass_3_1(self, inputs, n, numOut, name='hourglass_all'):
        """ Hourglass Module
    	Args:
    		inputs	: Input Tensor
    		n		: Number of downsampling step
    		numOut	: Number of Output Features (channels)
    		name	: Name of the block
    	"""
        with tf.name_scope(name):
            # Upper Branch
            up_1 = self._residual(inputs, numOut, name='up_1')
            # Lower Branch
            low_ = tf.contrib.layers.max_pool2d(inputs, [2, 2], [2, 2], padding='VALID')
            low_1 = self._residual(low_, numOut, name='low_1')

            if n > 0:
                low_2 = self._hourglass_3_2(low_1, n - 1, numOut, name='low_2')
            else:
                low_2 = self._residual(low_1, numOut, name='low_2')

            low_3 = self._residual(low_2, numOut, name='low_3')
            up_2 = tf.image.resize_nearest_neighbor(low_3, tf.shape(low_3)[1:3] * 2, name='upsampling')
            if self.modif:
                # Use of RELU
                return tf.nn.relu(tf.add_n([up_2, up_1]))
            else:
                return tf.add_n([up_2, up_1], name='out_hg')

    def _hourglass_3_(self, inputs, n, numOut, name='hourglass_all'):
        """ Hourglass Module
    	Args:
    		inputs	: Input Tensor
    		n		: Number of downsampling step
    		numOut	: Number of Output Features (channels)
    		name	: Name of the block
    	"""
        with tf.name_scope(name):
            # Upper Branch
            up_1 = self._residual(inputs, numOut, name='up_1')
            # Lower Branch
            low_ = tf.contrib.layers.max_pool2d(inputs, [2, 2], [2, 2], padding='VALID')
            low_1 = self._residual(low_, numOut, name='low_1')

            if n > 0:
                low_2 = self._hourglass_3_1(low_1, n - 1, numOut, name='low_2')
            else:
                low_2 = self._residual(low_1, numOut, name='low_2')

            low_3 = self._residual(low_2, numOut, name='low_3')
            up_2 = tf.image.resize_nearest_neighbor(low_3, tf.shape(low_3)[1:3] * 2, name='upsampling')
            if self.modif:
                # Use of RELU
                return tf.nn.relu(tf.add_n([up_2, up_1]))
            else:
                return tf.add_n([up_2, up_1], name='out_hg')

    def _hourglass_2_4(self, inputs, n, numOut, name='hourglass_all'):
        """ Hourglass Module
        Args:
            inputs	: Input Tensor
            n		: Number of downsampling step
            numOut	: Number of Output Features (channels)
            name	: Name of the block
        """
        with tf.name_scope(name):
            # Upper Branch
            up_1 = self._residual(inputs, numOut, name='up_1')
            # Lower Branch
            low_ = tf.contrib.layers.max_pool2d(inputs, [2, 2], [2, 2], padding='VALID')
            low_1 = self._residual(low_, numOut, name='low_1')

            if n > 0:
                low_2 = self._hourglass_2(low_1, n - 1, numOut, name='low_2')
            else:
                low_2 = self._residual(low_1, numOut, name='low_2')

            low_3 = self._residual(low_2, numOut, name='low_3')
            up_2 = tf.image.resize_nearest_neighbor(low_3, tf.shape(low_3)[1:3] * 2, name='upsampling')
            if self.modif:
                # Use of RELU
                return tf.nn.relu(tf.add_n([up_2, up_1]), name='out_hg')
            else:
                return tf.add_n([up_2, up_1], name='out_hg')

    def _hourglass_2_3(self, inputs, n, numOut, name='hourglass_all'):
        """ Hourglass Module
        Args:
            inputs	: Input Tensor
            n		: Number of downsampling step
            numOut	: Number of Output Features (channels)
            name	: Name of the block
        """
        with tf.name_scope(name):
            # Upper Branch
            up_1 = self._residual(inputs, numOut, name='up_1')
            # Lower Branch
            low_ = tf.contrib.layers.max_pool2d(inputs, [2, 2], [2, 2], padding='VALID')
            low_1 = self._residual(low_, numOut, name='low_1')

            if n > 0:
                low_2 = self._hourglass_2_4(low_1, n - 1, numOut, name='low_2')
            else:
                low_2 = self._residual(low_1, numOut, name='low_2')

            low_3 = self._residual(low_2, numOut, name='low_3')
            up_2 = tf.image.resize_nearest_neighbor(low_3, tf.shape(low_3)[1:3] * 2, name='upsampling')
            if self.modif:
                # Use of RELU
                return tf.nn.relu(tf.add_n([up_2, up_1]), name='out_hg')
            else:
                print('add+One_stage[3]:', self.One_stage[3])
                return tf.add_n([up_2, up_1,self.One_stage[3]], name='out_hg')

    def _hourglass_2_2(self, inputs, n, numOut, name='hourglass_all'):
        """ Hourglass Module
        Args:
            inputs	: Input Tensor
            n		: Number of downsampling step
            numOut	: Number of Output Features (channels)
            name	: Name of the block
        """
        with tf.name_scope(name):
            # Upper Branch
            up_1 = self._residual(inputs, numOut, name='up_1')
            # Lower Branch
            low_ = tf.contrib.layers.max_pool2d(inputs, [2, 2], [2, 2], padding='VALID')
            low_1 = self._residual(low_, numOut, name='low_1')

            if n > 0:
                low_2 = self._hourglass_2_3(low_1, n - 1, numOut, name='low_2')
            else:
                low_2 = self._residual(low_1, numOut, name='low_2')

            low_3 = self._residual(low_2, numOut, name='low_3')
            up_2 = tf.image.resize_nearest_neighbor(low_3, tf.shape(low_3)[1:3] * 2, name='upsampling')
            if self.modif:
                # Use of RELU
                return tf.nn.relu(tf.add_n([up_2, up_1]), name='out_hg')
            else:
                print('add+One_stage[2]:', self.One_stage[2])
                return tf.add_n([up_2, up_1,self.One_stage[2]], name='out_hg')

    def _hourglass_2_1(self, inputs, n, numOut, name='hourglass_all'):
        """ Hourglass Module
        Args:
            inputs	: Input Tensor
            n		: Number of downsampling step
            numOut	: Number of Output Features (channels)
            name	: Name of the block
        """
        with tf.name_scope(name):
            # Upper Branch
            up_1 = self._residual(inputs, numOut, name='up_1')
            # Lower Branch
            low_ = tf.contrib.layers.max_pool2d(inputs, [2, 2], [2, 2], padding='VALID')
            low_1 = self._residual(low_, numOut, name='low_1')

            if n > 0:
                low_2 = self._hourglass_2_2(low_1, n - 1, numOut, name='low_2')
            else:
                low_2 = self._residual(low_1, numOut, name='low_2')

            low_3 = self._residual(low_2, numOut, name='low_3')
            up_2 = tf.image.resize_nearest_neighbor(low_3, tf.shape(low_3)[1:3] * 2, name='upsampling')
            if self.modif:
                # Use of RELU
                return tf.nn.relu(tf.add_n([up_2, up_1]), name='out_hg')
            else:
                print('add+One_stage[1]:', self.One_stage[1])
                return tf.add_n([up_2, up_1,self.One_stage[1]], name='out_hg')

    def _hourglass_2_(self, inputs, n, numOut, name='hourglass_all'):
        """ Hourglass Module
    	Args:
    		inputs	: Input Tensor
    		n		: Number of downsampling step
    		numOut	: Number of Output Features (channels)
    		name	: Name of the block
    	"""
        with tf.name_scope(name):
            # Upper Branch
            up_1 = self._residual(inputs, numOut, name='up_1')
            # Lower Branch
            low_ = tf.contrib.layers.max_pool2d(inputs, [2, 2], [2, 2], padding='VALID')
            low_1 = self._residual(low_, numOut, name='low_1')

            if n > 0:
                low_2 = self._hourglass_2_1(low_1, n - 1, numOut, name='low_2')
            else:
                low_2 = self._residual(low_1, numOut, name='low_2')

            low_3 = self._residual(low_2, numOut, name='low_3')
            up_2 = tf.image.resize_nearest_neighbor(low_3, tf.shape(low_3)[1:3] * 2, name='upsampling')
            if self.modif:
                # Use of RELU
                return tf.nn.relu(tf.add_n([up_2, up_1]), name='out_hg')
            else:
                print('add+One_stage[0]:', self.One_stage[0])
                return tf.add_n([up_2, up_1,self.One_stage[0]], name='out_hg')

    def _argmax(self, tensor):
        """ ArgMax
		Args:
			tensor	: 2D - Tensor (Height x Width : 64x64 )
		Returns:
			arg		: Tuple of max position
		"""
        resh = tf.reshape(tensor, [-1])
        argmax = tf.arg_max(resh, 0)
        return (argmax // tensor.get_shape().as_list()[0], argmax % tensor.get_shape().as_list()[0])

    def _compute_err(self, u, v):
        """ Given 2 tensors compute the euclidean distance (L2) between maxima locations
		Args:
			u		: 2D - Tensor (Height x Width : 64x64 )
			v		: 2D - Tensor (Height x Width : 64x64 )
		Returns:
			(float) : Distance (in [0,1])
		"""
        u_x, u_y = self._argmax(u)
        v_x, v_y = self._argmax(v)
        return tf.divide(tf.sqrt(tf.square(tf.to_float(u_x - v_x)) + tf.square(tf.to_float(u_y - v_y))),
                         tf.to_float(91))

    def _accur(self, pred, gtMap, num_image):
        """ Given a Prediction batch (pred) and a Ground Truth batch (gtMaps),
		returns one minus the mean distance.
		Args:
			pred		: Prediction Batch (shape = num_image x 64 x 64)
			gtMaps		: Ground Truth Batch (shape = num_image x 64 x 64)
			num_image 	: (int) Number of images in batch
		Returns:
			(float)
		"""
        err = tf.to_float(0)
        for i in range(num_image):
            err = tf.add(err, self._compute_err(pred[i], gtMap[i]))
        return tf.subtract(tf.to_float(1), err / num_image)

    # MULTI CONTEXT ATTENTION MECHANISM
    # WORK IN PROGRESS DO NOT USE THESE METHODS
    # BASED ON:
    # Multi-Context Attention for Human Pose Estimation
    # Authors: Xiao Chu, Wei Yang, Wanli Ouyang, Cheng Ma, Alan L. Yuille, Xiaogang Wang
    # Paper: https://arxiv.org/abs/1702.07432
    # GitHub Torch7 Code: https://github.com/bearpaw/pose-attention

    def _bn_relu(self, inputs):
        norm = tf.contrib.layers.batch_norm(inputs, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu,
                                            is_training=self.training)
        return norm

    def _pool_layer(self, inputs, numOut, name='pool_layer'):
        with tf.name_scope(name):
            bnr_1 = self._bn_relu(inputs)
            pool = tf.contrib.layers.max_pool2d(bnr_1, [2, 2], [2, 2], padding='VALID')
            pad_1 = tf.pad(pool, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]))
            conv_1 = self._conv(pad_1, numOut, kernel_size=3, strides=1, name='conv')
            bnr_2 = self._bn_relu(conv_1)
            pad_2 = tf.pad(bnr_2, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]))
            conv_2 = self._conv(pad_2, numOut, kernel_size=3, strides=1, name='conv')
            upsample = tf.image.resize_nearest_neighbor(conv_2, tf.shape(conv_2)[1:3] * 2, name='upsampling')
        return upsample

    def _attention_iter(self, inputs, lrnSize, itersize, name='attention_iter'):
        with tf.name_scope(name):
            numIn = inputs.get_shape().as_list()[3]
            padding = np.floor(lrnSize / 2)
            pad = tf.pad(inputs, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]))
            U = self._conv(pad, filters=1, kernel_size=3, strides=1)
            pad_2 = tf.pad(U, np.array([[0, 0], [padding, padding], [padding, padding], [0, 0]]))
            sharedK = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)([lrnSize, lrnSize, 1, 1]),
                                  name='shared_weights')
            Q = []
            C = []
            for i in range(itersize):
                if i == 0:
                    conv = tf.nn.conv2d(pad_2, sharedK, [1, 1, 1, 1], padding='VALID', data_format='NHWC')
                else:
                    conv = tf.nn.conv2d(Q[i - 1], sharedK, [1, 1, 1, 1], padding='SAME', data_format='NHWC')
                C.append(conv)
                Q_tmp = tf.nn.sigmoid(tf.add_n([C[i], U]))
                Q.append(Q_tmp)
            stacks = []
            for i in range(numIn):
                stacks.append(Q[-1])
            pfeat = tf.multiply(inputs, tf.concat(stacks, axis=3))
        return pfeat

    def _attention_part_crf(self, inputs, lrnSize, itersize, usepart, name='attention_part'):
        with tf.name_scope(name):
            if usepart == 0:
                return self._attention_iter(inputs, lrnSize, itersize)
            else:
                partnum = self.outDim
                pre = []
                for i in range(partnum):
                    att = self._attention_iter(inputs, lrnSize, itersize)
                    pad = tf.pad(att, np.array([[0, 0], [0, 0], [0, 0], [0, 0]]))
                    s = self._conv(pad, filters=1, kernel_size=1, strides=1)
                    pre.append(s)
                return tf.concat(pre, axis=3)

    def _residual_pool(self, inputs, numOut, name='residual_pool'):
        with tf.name_scope(name):
            return tf.add_n(
                [self._conv_block(inputs, numOut), self._skip_layer(inputs, numOut), self._pool_layer(inputs, numOut)])

    def _rep_residual(self, inputs, numOut, nRep, name='rep_residual'):
        with tf.name_scope(name):
            out = [None] * nRep
            for i in range(nRep):
                if i == 0:
                    tmpout = self._residual(inputs, numOut)
                else:
                    tmpout = self._residual_pool(out[i - 1], numOut)
                out[i] = tmpout
            return out[nRep - 1]

    def _hg_mcam(self, inputs, n, numOut, imSize, nModual, name='mcam_hg'):
        with tf.name_scope(name):
            # ------------Upper Branch
            pool = tf.contrib.layers.max_pool2d(inputs, [2, 2], [2, 2], padding='VALID')
            up = []
            low = []
            for i in range(nModual):
                if i == 0:
                    if n > 1:
                        tmpup = self._rep_residual(inputs, numOut, n - 1)
                    else:
                        tmpup = self._residual(inputs, numOut)
                    tmplow = self._residual(pool, numOut)
                else:
                    if n > 1:
                        tmpup = self._rep_residual(up[i - 1], numOut, n - 1)
                    else:
                        tmpup = self._residual_pool(up[i - 1], numOut)
                    tmplow = self._residual(low[i - 1], numOut)
                up.append(tmpup)
                low.append(tmplow)
            # up[i] = tmpup
            # low[i] = tmplow
            # ----------------Lower Branch
            if n > 1:
                low2 = self._hg_mcam(low[-1], n - 1, numOut, int(imSize / 2), nModual)
            else:
                low2 = self._residual(low[-1], numOut)
            low3 = self._residual(low2, numOut)
            up_2 = tf.image.resize_nearest_neighbor(low3, tf.shape(low3)[1:3] * 2, name='upsampling')
            return tf.add_n([up[-1], up_2], name='out_hg')

    def _lin(self, inputs, numOut, name='lin'):
        l = self._conv(inputs, filters=numOut, kernel_size=1, strides=1)
        return self._bn_relu(l)

    ##此沙漏网络模型，是加了"注意力机制"功能
    def _graph_mcam(self, inputs):
        with tf.name_scope('preprocessing'):
            pad1 = tf.pad(inputs, np.array([[0, 0], [3, 3], [3, 3], [0, 0]]))
            cnv1_ = self._conv(pad1, filters=64, kernel_size=7, strides=1)
            cnv1 = self._bn_relu(cnv1_)
            r1 = self._residual(cnv1, 64)
            pool1 = tf.contrib.layers.max_pool2d(r1, [2, 2], [2, 2], padding='VALID')
            r2 = self._residual(pool1, 64)
            r3 = self._residual(r2, 128)
            pool2 = tf.contrib.layers.max_pool2d(r3, [2, 2], [2, 2], padding='VALID')
            r4 = self._residual(pool2, 128)
            r5 = self._residual(r4, 128)
            r6 = self._residual(r5, 256)  ##我计算了下：r6 的输出维度为 64×64×256
        out = []
        inter = []
        inter.append(r6)
        if self.nLow == 3:
            nModual = int(16 / self.nStack)
        else:
            nModual = int(8 / self.nStack)
        with tf.name_scope('stacks'):
            for i in range(self.nStack):
                with tf.name_scope('houglass_' + str(i + 1)):
                    hg = self._hg_mcam(inter[i], self.nLow, self.nFeat, 64, nModual)

                if i == self.nStack - 1:
                    ll1 = self._lin(hg, self.nFeat * 2)
                    ll2 = self._lin(ll1, self.nFeat * 2)
                    drop = tf.layers.dropout(ll2, rate=0.1, training=self.training)
                    att = self._attention_part_crf(drop, 1, 3, 0)
                    tmpOut = self._attention_part_crf(att, 1, 3, 1)
                else:
                    ll1 = self._lin(hg, self.nFeat)
                    ll2 = self._lin(ll1, self.nFeat)
                    drop = tf.layers.dropout(ll2, rate=0.1, training=self.training)
                    if i > self.nStack // 2:
                        att = self._attention_part_crf(drop, 1, 3, 0)
                        tmpOut = self._attention_part_crf(att, 1, 3, 1)
                    else:
                        att = self._attention_part_crf(ll2, 1, 3, 0)
                        tmpOut = self._conv(att, filters=self.outDim, kernel_size=1, strides=1)
                out.append(tmpOut)
                if i < self.nStack - 1:
                    outmap = self._conv(tmpOut, filters=self.nFeat, kernel_size=1, strides=1)
                    ll3 = self._lin(outmap, self.nFeat)
                    tmointer = tf.add_n([inter[i], outmap, ll3])
                    inter.append(tmointer)
        return tf.stack(out, axis=1, name='final_output')
