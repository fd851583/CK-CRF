# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 16:39:58 2020

@author: Xiuxiu
"""

import os, random
import numpy as np
import cv2 as cv
import tensorflow as tf

def read_list(image_dir, label_dir):
    images = os.listdir(image_dir)
    labels = os.listdir(label_dir)
    images = [image_dir+image for image in images]
    labels = [label_dir+label for label in labels]
    images.sort()
    labels.sort()
    
    output = np.vstack((images, labels))
    output = np.transpose(output)
    output = np.reshape(output, [104, 2])
    
    return output

def read_little_sample(file_list, image_dir, label_dir, sample_width=80, sample_height=80, batch_size=2):
    image_output = []
    label_output = []
    
    for i in range(batch_size):
        loca1 = random.randint(0, len(file_list)-1)
        img = cv.imread(image_dir + file_list[loca1])
        lab = cv.imread(label_dir + file_list[loca1], 0)
        lab = cv.GaussianBlur(lab, (11,11), 0)
        img_shape = img.shape
        
        width_location = random.randint(0, (img_shape[1]-sample_width))
        height_location = random.randint(0, (img_shape[0]-sample_height))
        img_sample = img[height_location:height_location+sample_height, width_location:width_location+sample_width, :]
        lab_sample = lab[height_location:height_location+sample_height, width_location:width_location+sample_width]
        
        image_output.append(img_sample)
        label_output.append(lab_sample)
        
    image_output = np.reshape(image_output, [batch_size, sample_height, sample_width, 3])/255
    label_output = np.reshape(label_output, [batch_size, sample_height, sample_width, 1])/255
            
    return image_output, label_output

def read_little_sample_v1(file_list, image_dir, label_dir, sample_width=80, sample_height=80, batch_size=2):
    image_output = []
    label_output = []
    
    for i in range(batch_size):
        output_flag = 0
        while output_flag == 0:
            loca1 = random.randint(0, len(file_list)-1)
            img = cv.imread(image_dir + file_list[loca1])
            lab = cv.imread(label_dir + file_list[loca1], 0)
            lab = cv.GaussianBlur(lab, (11,11), 0)
            img_shape = img.shape
            
            width_location = random.randint(0, (img_shape[1]-sample_width))
            height_location = random.randint(0, (img_shape[0]-sample_height))
            img_sample = img[height_location:height_location+sample_height, width_location:width_location+sample_width, :]
            lab_sample = lab[height_location:height_location+sample_height, width_location:width_location+sample_width]
            
            if np.sum(lab_sample) > int((sample_width*sample_height) * 255 * 0.1):
                output_flag = 1
        
        image_output.append(img_sample)
        label_output.append(lab_sample)
        
    image_output = np.reshape(image_output, [batch_size, sample_height, sample_width, 3])/255
    label_output = np.reshape(label_output, [batch_size, sample_height, sample_width, 1])/255
            
    return image_output, label_output

def read_test(file_name, image_dir, label_dir): 

    img = cv.imread(image_dir + file_name)
    lab = cv.imread(label_dir + file_name, 0)
        
    img = np.reshape(img, [1, 128, 128, 3])/255
    lab = np.reshape(lab, [1, 128, 128, 1])/255
    
    return img, lab

def imgoverly(image, label, output):
    shape = image.shape
    temp3 = np.zeros_like(image)
    label = np.reshape(label, [label.shape[0], label.shape[1]])
    temp3[:, :, 2] = label
    temp1 = cv.addWeighted(image, 1, temp3, 0.7, 0)
    output = np.reshape(output, [output.shape[0], output.shape[1]])
    temp3[:, :, 2] = output
    temp2 = cv.addWeighted(image, 1, temp3, 0.7, 0)
    result = np.zeros([shape[0], shape[1]*2, 3])
    result[:, :shape[1], :] = temp1
    result[:, shape[1]:, :] = temp2
    
    return result

def symmetry(img, mode=0):
    new_img = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if mode == 0:
                new_img[i, j] = img[img.shape[0]-i-1, j]
            elif mode == 1:
                new_img[i, j] = img[i, img.shape[1]-j-1]
            elif mode == 2:
                new_img[i, j] = img[img.shape[0]-i-1, img.shape[1]-j-1]
    
    return new_img

def symmetry1(img, mode=0):
    if mode == 0:
        return img
    elif mode == 1:
        new_img = np.zeros_like(img)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                new_img[i, j] = img[i, img.shape[1]-j-1]
        return new_img

def compute_result(input1):
    sum1 = np.sum(input1)
    output = sum1/(input1.shape[0]*input1.shape[1]*input1.shape[2]*input1.shape[3])
    
    return output

def compute_IoU(p, l, num=0.5):
    p = tf.where(p>num, tf.ones_like(p), tf.zeros_like(p))
    l = tf.where(l>num, tf.ones_like(l), tf.zeros_like(l))
    temp1 = tf.where(tf.equal(p, 0), tf.ones_like(p), tf.zeros_like(p))
    temp2 = tf.where(tf.equal(l, 1), tf.ones_like(p), tf.zeros_like(p))
    temp3 = tf.multiply(temp1, temp2)
    
    temp4 = tf.where(tf.equal(p, 1), tf.ones_like(p), tf.zeros_like(p))
    temp5 = tf.where(tf.equal(l, 0), tf.ones_like(p), tf.zeros_like(p))
    temp6 = tf.multiply(temp4, temp5)
    
    #temp7 = tf.where(tf.equal(p, l), tf.ones_like(p), tf.zeros_like(p))
    temp7 = tf.multiply(p, l)
    
    temp8 = tf.reduce_sum(temp3)
    temp9 = tf.reduce_sum(temp6)
    temp10 = tf.reduce_sum(temp7)
    output = temp10/(temp8+temp9+temp10)
    
    return output

def compute_accuracy(p, l, num=0.5):
    p = tf.where(p>num, tf.ones_like(p), tf.zeros_like(p))
    l = tf.where(l>num, tf.ones_like(l), tf.zeros_like(l))
    
    temp1 = tf.where(tf.equal(p, l), tf.ones_like(p), tf.zeros_like(p))
    temp2 = tf.reduce_sum(temp1)
    
    temp3 = tf.ones_like(p)
    temp4 = tf.reduce_sum(temp3)
    
    return temp2/temp4

def compute_precision(p, l, num=0.5):
    p = tf.where(p>num, tf.ones_like(p), tf.zeros_like(p))
    l = tf.where(l>num, tf.ones_like(l), tf.zeros_like(l))
    
    temp1 = tf.multiply(p, l)
    temp2 = tf.reduce_sum(temp1)
    
    temp3 = tf.reduce_sum(l)
    
    return temp2/temp3

def compute_recall(p, l, num=0.5):
    p = tf.where(p>num, tf.ones_like(p), tf.zeros_like(p))
    l = tf.where(l>num, tf.ones_like(l), tf.zeros_like(l))
    
    temp1 = tf.multiply(p, l)
    temp2 = tf.reduce_sum(temp1)
    
    temp3 = tf.reduce_sum(p)
    
    return temp2/temp3

def compute_accuracy1(p, l, num=0.5):
    p = tf.where(p>num, tf.ones_like(p), tf.zeros_like(p))
    l = tf.where(l>num, tf.ones_like(l), tf.zeros_like(l))
    
    temp1 = tf.multiply(p, l)
    temp2 = tf.reduce_sum(temp1)
    temp3 = tf.reduce_sum(l)
    
    return temp2/temp3
    

def change1(image, label):
    rand1 = random.randint(0,1)
    if rand1 == 0:
        angle_rand = random.randint(-360, 360)
        x1_rand = random.randint(412, 612)
        y1_rand = random.randint(206, 306)
        M1 = cv.getRotationMatrix2D((x1_rand, y1_rand), angle_rand, 1)
        image = cv.warpAffine(image, M1, (image.shape[1], image.shape[0]))
        label = cv.warpAffine(label, M1, (label.shape[1], label.shape[0]))
    
    rand2 = random.randint(0,1)
    if rand2 == 0:
        x2_rand = random.randint(-400, 400)
        y2_rand = random.randint(-200, 200)
        M2 = np.float32([[1, 0, x2_rand], [0, 1, y2_rand]])
        image = cv.warpAffine(image, M2, (image.shape[1], image.shape[0]))
        label = cv.warpAffine(label, M2, (label.shape[1], label.shape[0]))
    
    return image, label