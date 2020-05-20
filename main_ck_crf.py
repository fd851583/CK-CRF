# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 16:34:14 2020

@author: Xiuxiu
"""

import tensorflow as tf
import cv2 as cv
import numpy as np
import reader, lossStore, netStore, os
tf.reset_default_graph()

train_dir1 = "E:/pyground/city_test1/dataset/train/image/"
train_dir2 = "E:/pyground/city_test1/dataset/train/label/"
test_dir1 = "E:/pyground/city_test1/dataset/test/image/"
test_dir2 = "E:/pyground/city_test1/dataset/test/label/"

test_dir3 = "E:/pyground/city_test1/test_dataset/image/"
test_dir4 = "E:/pyground/city_test1/test_dataset/label/"

train_list = os.listdir(train_dir2)
test_list = os.listdir(test_dir2)
test_list1 = os.listdir(test_dir4)

batch_size = 2
train_times = 10000
test_times = 1000
total_batch = int(len(train_list)/batch_size)

image = tf.placeholder(tf.float32, [None, 128, 128, 3])
label = tf.placeholder(tf.float32, [None, 128, 128, 1])

output = netStore.net3(image, "net0")

label1 = tf.where(label>0.5, tf.ones_like(label), tf.zeros_like(label))

output1 = tf.where(output>0.5, tf.ones_like(output), tf.zeros_like(output))

crf_output = netStore.crf_net(output, 3, name="crf")

loss_pix = lossStore.pix_loss(output, label)

loss_crf = lossStore.crf_loss1(crf_output)

loss = tf.reduce_mean(loss_pix + loss_crf)

IoU = reader.compute_IoU(output, label)

acc = reader.compute_accuracy(output, label)

prec = reader.compute_precision(output, label)

recall = reader.compute_recall(output, label)

output1 = tf.where(output>0.5, tf.ones_like(output), tf.zeros_like(output))

global_step = tf.Variable(0, trainable=False)

lr1 = tf.train.exponential_decay(0.001, global_step, 1000, 0.9, staircase=True)

all_vars = tf.trainable_variables()
net_vars = [var for var in all_vars if "net0" in var.name]

net0_optimizer = tf.train.AdamOptimizer(lr1).minimize(loss_pix, var_list = net_vars, global_step=global_step)
net0_optimizer1 = tf.train.AdamOptimizer(lr1).minimize(loss, var_list = net_vars, global_step=global_step)

saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

tf.summary.scalar('IoU', IoU)
tf.summary.scalar('loss_pix', loss_pix)

with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
    ckpt = tf.train.get_checkpoint_state('./models/')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Import models successful!')
    else:
        sess.run(tf.global_variables_initializer())
        print('Initialize successful!')
        
    merged_summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter("./summary/", sess.graph)
        
    print("Training Start!")
    IoU_list = []
    for i in range(train_times):

        image_data, label_data = reader.read_little_sample_v1(train_list, train_dir1, train_dir2, 128, 128, 10)
        feeds = {image:image_data, label:label_data}
        if i < 500:
            _, summary_str, gt = sess.run([net0_optimizer, merged_summary_op, global_step], feeds)
        else:
            _, summary_str, gt = sess.run([net0_optimizer1, merged_summary_op, global_step], feeds)
        summary_writer.add_summary(summary_str, gt)
        
        if i % 250 == 0:           
            img, lab, out, loss1, IoU_output = sess.run([image, label, output, loss_pix, IoU], feeds)
            print("Train! itor:" + str(i) + " loss:" + str(loss1), str(IoU_output))
            cv.imwrite('./train_record/' + str(i) + '_image' + '.png' , img[0]*255)
            cv.imwrite('./train_record/' + str(i) + '_label' + '.png' , lab[0]*255)
            cv.imwrite('./train_record/' + str(i) + '_output_' + str(IoU_output) + '.png' , out[0]*255)
            
            saver.save(sess, "./models/road_model.cpkt", global_step=global_step)
    saver.save(sess, "./models/road_model.cpkt", global_step=global_step)
    print("Trainning Finish!")
    
#Test plan 1
#    print("Testing Start!")
#    IoU_sum = 0
#    for i in range(test_times):
#
#        image_data, label_data = reader.read_little_sample_v1(test_list, test_dir1, test_dir2, 128, 128, 1)
#        feeds = {image:image_data, label:label_data}
#           
#        img, lab, out, IoU_output = sess.run([image, label, output, IoU], feeds)
##        print("Train! itor:" + str(i) + " IoU:", str(IoU_output))
##        if i % 50 == 0:
##            cv.imwrite('./test_record/' + str(i) + '_image' + '.png' , img[0]*255)
##            cv.imwrite('./test_record/' + str(i) + '_label' + '.png' , lab[0]*255)
##            cv.imwrite('./test_record/' + str(i) + '_output' + '.png' , out[0]*255)
#        IoU_sum += IoU_output
#    
#    IoU_result = IoU_sum/test_times
#    print("Testing Finish!")
#    print("IoU:", str(IoU_result))

#Test plan 2
    print("Start Testing!")
    IoU_sum = 0
    acc_sum = 0
    prec_sum = 0
    recall_sum = 0
    for i in range(len(test_list1)):
        image_data, label_data = reader.read_test(test_list1[i], test_dir3, test_dir4)
        feeds = {image:image_data, label:label_data}
        img, lab, out, loss1, IoU_output, acc_output, prec_output, recall_output = sess.run([image, label1, output1, loss_pix, IoU, acc, prec, recall], feeds)
        print("Test! ", test_list1[i], str(IoU_output))
        IoU_sum += IoU_output
        acc_sum += acc_output
        prec_sum += prec_output
        recall_sum += recall_output

        cv.imwrite('./test_record/' + test_list1[i][:-4] + '_image' + '.png' , img[0]*255)
        cv.imwrite('./test_record/' + test_list1[i][:-4] + '_label' + '.png' , lab[0]*255)
        cv.imwrite('./test_record/' + test_list1[i][:-4] + '_output' + '.png' , out[0]*255)
        
    sum_result = IoU_sum/(len(test_list1))
    acc_result = acc_sum/(len(test_list1))
    prec_result = prec_sum/(len(test_list1))
    recall_result = recall_sum/(len(test_list1))
    print("Testing Finish! Test result:", str(sum_result), str(acc_result), str(prec_result), str(recall_result))