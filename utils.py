#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 23:12:25 2020

@author: bigmao
"""
import numpy as np
import os
from scipy import io
from scipy import interpolate
import pyprind

def frame_info_pool(sw_list):
    
    '''
    This function do the following work:
        1. calculates the start_frame and the end_frame in 30 Hz
        2. calculates the frame length for generating mask
        3. get the max frame length
    '''
    
    total_len = []
    
    for ind in sw_list.keys():
    
        item = sw_list[ind]
        
        # read original signal
        #item['fnames']
        
        # get start frame and end frame of the swallow
        item['start_frame_30'] = np.int(np.ceil(item['start_frame']/2))
        item['end_frame_30'] = np.int(np.ceil(item['end_frame']/2))
        
        temp_len = item['end_frame_30']-item['start_frame_30']+1
        
        total_len.append(temp_len)
        
        item['frame_len'] = temp_len

        
    return sw_list, max(total_len)

def update_signal(sw_list,max_len,prefix = 'signal', fs_ori = 4000,frame_f = 30,sample_factor = 128):
    
    
    '''
    This function updates the signal for each swallow recording. In details, it implements:
        1. read the original signal in 4000hz
        2. re-sample the signal accroding to sample_factor
        3. segment the signal
        4. pad zeros for the signal
        5. mask generation for signal
        
    inputs:
        sw_list: a dictionary, each item is a swallow recording. The start and end points at 30Hz frame must be calculated first
        max_len: the max length of the signal. Note that this length is calculated based on frame frequency (30Hz)
        prefix: the folder whicn contains the mat files.
        fs_ori: the original sampling rate of the signal, which is 4000
        frame_f: the frequency of the frame, mush be 30hz
        sample_factor: how many time points of signal in each frame
    '''

    # calculate new frequency
    fs_new = sample_factor*frame_f
    sample_num = len(sw_list)
    bar = pyprind.ProgBar(sample_num, monitor=True)
    # now segment the signal for each swallow
    for ind in sw_list.keys():
    
        item = sw_list[ind]
    
        # check whether the item is a dictionary
        if type(item).__name__ != 'dict':
            raise ValueError('the input is not a dict')
            
        # get the file name of the swallow
        file_name = item['fnames']
        
        # get the file path
        file_path=os.path.join(prefix,item['participant'].lower(),file_name+'.mat')
        
        # chech the path
        if not os.path.exists(file_path):
            raise FileNotFoundError('the signal file %s cannot be found'%file_path)
            
        # read the mat file
        data_old=io.loadmat(file_path)
        
        # the python read mat file as a dictionary, we need the key name without '__'
        useful_name = [key for key in data_old.keys() if key[0]!='_']
        
        # check whether it has two keys
        if len(useful_name)==1:
            useful_name = useful_name[0]
        else:
            raise KeyError('the signal mat file %s has two keys'%file_path)
         
        # get the entire signal
        signal_whole = data_old[useful_name]
        
        # get the signal length of the entire signal
        num,_=np.shape(signal_whole)
        
        # old time index
        t_old=np.array([i/fs_ori for i in range(num)])
        
        # inerpolate
        f = interpolate.interp1d(t_old, signal_whole.T)
        
        # get new tiem index after dawnsampling
        t_new=np.arange(0, t_old[-1], 1/fs_new)
        
        data_new=f(t_new).T
        
        # segment the signal
        start_time_point=(item['start_frame_30']-1)*sample_factor
        end_time_point=(item['end_frame_30'])*sample_factor # slice doesn't consider the last element!!!!!
        
        temp_signal = np.float32(data_new[start_time_point:end_time_point])
        
        # get signal length
        signal_len,_ = temp_signal.shape
        
        
        # check, whether the singal length is a integer multiple of the frame length
        if not signal_len/sample_factor == item['frame_len']:
            raise ValueError('the signal length is not an integer multiple of the frame length')
        
        # pad the signal
        signal_pad = np.zeros([max_len*sample_factor,4],np.float32)
        signal_pad[:signal_len,:] = temp_signal
        
        # generate mask
        temp_mask = np.zeros([max_len*sample_factor,1],np.float32)
        temp_mask[:signal_len,:]=1.0
        
        item['signal_len'] = signal_len
        item['signal_seg'] = signal_pad
        item['mask'] = temp_mask
        
        bar.update()
    print(bar)  
    return sw_list


def get_data_pool(sw_list):
    '''
    This function forms the signals and masks from all the recordings to numpy arrays 
    '''
    signal_pool = []
    mask_pool = []
    
    
    
    for ind in sw_list.keys():
    
        item = sw_list[ind]
    
        signal_pool.append(item['signal_seg'])
        mask_pool.append(item['mask'])
    
    signal_pool = np.array(signal_pool)
    mask_pool = np.array(mask_pool)
    
    return signal_pool,mask_pool