'''

train each level with stacks

'''
import tensorflow as tf
import os

import numpy as np
import matplotlib.pyplot as plt

import pickle

from stacked_AE import ae_hie



def dis_loss_pair(x_in,x_out):
    x_ch = x_in.shape[-1]
   # mask = tf.slice(x_in, [0, 0, x_ch-1], [-1, -1, -1])
    x_sig = tf.slice(x_in, [0, 0, 0], [-1, -1, x_ch-1])
    

    x_out_sig = tf.slice(x_out, [0, 0, 0], [-1, -1, x_ch-1])   
    
    nsr = tf.reduce_sum((x_sig-x_out_sig)**2,1)/tf.reduce_sum(x_sig**2,1)
    
    d_loss = tf.reduce_mean(nsr)
    
    
    return d_loss






def apply_grads_L(x,level):
        
    with tf.GradientTape() as tape:
        dic = auto_hie(x)
        diff_loss = dis_loss_pair(dic['la_l%s'%level],dic['l%s_int'%level])
    
    name_ls = ['cen_l%s'%level,'cde_l%s'%level]
    var = [item for item in auto_hie.trainable_variables if any(st in name_ls for st in item.name.split('/'))]
    grad= tape.gradient(diff_loss, var)
    optimizer.apply_gradients(zip(grad, var))
    



def train_step_L(level):
    @tf.function
    def distributed_apply_grads_L(dataset_inputs):
        strategy.run(apply_grads_L, args=(dataset_inputs,level))

    return distributed_apply_grads_L


def eva_loop(r_data,level):
    ep_loss = []
    for x_data in r_data: 
        temp_dic = auto_hie(x_data)
        temp_loss = dis_loss_pair(temp_dic['la_l%s'%level],temp_dic['l%s_int'%level])
        ep_loss.append(temp_loss)

    return np.mean(ep_loss)

def eva_loop_ft(r_data):
    ep_loss = []
    for x_data in r_data: 
        temp_dic = auto_hie(x_data)
        temp_loss = dis_loss_pair(x_data,temp_dic['l1_rec'])
        ep_loss.append(temp_loss)

    return np.mean(ep_loss)


def apply_grads_ft(x):
 
            

    with tf.GradientTape() as tape:
        dic = auto_hie(x)

        diff_loss =(dis_loss_pair(x,dic['l1_rec']) +
                    dis_loss_pair(dic['la_l2'],dic['l2_rec']) +
                    dis_loss_pair(dic['la_l3'],dic['l3_rec']) +
                    dis_loss_pair(dic['la_l4'],dic['l4_rec']) +
                    dis_loss_pair(dic['la_l5'],dic['l5_rec']) +
                    dis_loss_pair(dic['la_l6'],dic['l6_rec']) +
                    dis_loss_pair(dic['la_l7'],dic['l7_rec'])
                    )

    grad= tape.gradient(diff_loss, auto_hie.trainable_variables)
    ft_op.apply_gradients(zip(grad, auto_hie.trainable_variables)) 



def train_step_ft():
    @tf.function
    def distributed_apply_grads_ft(dataset_inputs):
        strategy.run(apply_grads_ft, args=(dataset_inputs,))

    return distributed_apply_grads_ft







def train_each_level(level,train_th,int_ep):
    with strategy.scope():
        save_path = os.path.join(save_folder,'L%d_loss_ep%d.pkl'%(level,int_ep))
        loss_index_L = eva_loop(r_data,level)

        
        
        if os.path.exists(save_path):
            loss_tracking = pickle.load(open(save_path, "rb" ))
            ep = (len(loss_tracking)-1)*int_ep+1

        else:
            loss_tracking = []
            ep = 0

        if loss_index_L<=train_th:
            print('the loss of loaded model at level_%d is %0.8f, which has already reached the threshold %0.8f'%(level,loss_index_L,train_th ))
        
        else:
            train_L = train_step_L(level=level)
                
            while loss_index_L > train_th:            
                
                for x_data in r_data: 
                    train_L(x_data)    
                    
                    
                if ep%int_ep==0:   
                    
                    temp_loss = eva_loop(r_data,level)
                    
                    print('level %d at epoch %d, loss is %0.8f, threshold is %0.8f'%(level, ep,temp_loss,train_th))
                    
                    loss_tracking.append(temp_loss)
            
                    raw_data = signal_pool[:10,:,:]
                    temp_dic = auto_hie(raw_data)
                          
                    plt.plot(np.log10(loss_tracking))
                    plt.xlabel('Epoch*%d'%int_ep)
                    plt.ylabel('log10(loss)')
                    file_name = os.path.join(save_folder,'L%d_loss.png'%level)
                    plt.savefig(file_name)
                    plt.close()
            
                    plt.plot(temp_dic['la_l%d'%level][1,:,2]);plt.plot(temp_dic['l%d_int'%level][1,:,2])
                    file_name = os.path.join(save_folder,'example_%d.png'%level)
                    plt.savefig(file_name)
                    plt.close()
                    plt.plot(temp_dic['la_l%d'%level][1,:200,2]);plt.plot(temp_dic['l%d_int'%level][1,:200,2])
                    file_name = os.path.join(save_folder,'example_zoomin_%d.png'%level)
                    plt.savefig(file_name)
                    plt.close()
                    
                    pickle.dump(loss_tracking, open(save_path, "wb" ) )
                    
                    
                    loss_index_L = temp_loss
                ep += 1  
            print('now training at level_%d finished when the threshould is %0.6f'%(level, train_th))




def train_each_level_ft(train_th,int_ep):
    loss_index_L = eva_loop_ft(r_data)
    save_path = os.path.join(save_folder,'ft_loss_ep%d.pkl'%int_ep)
    if os.path.exists(save_path):
        loss_tracking = pickle.load(open(save_path, "rb" ))
        ep = (len(loss_tracking)-1)*10+1

    else:
        loss_tracking = []
        ep = 0

    
    
    if loss_index_L<=train_th:
        print('the loss of loaded model at fine-tune level is %0.8f, which has already reached the threshold %0.8f'%(loss_index_L,train_th ))    
        
    else:
        train_ft = train_step_ft()
        
        while loss_index_L > train_th:        
            
            for x_data in r_data: 
                train_ft(x_data)    
                
                
            if ep%int_ep==0:   
                temp_loss = eva_loop_ft(r_data)
                print('Finetune at epoch %d, loss is %0.8f, threshold is %0.8f'%(ep,temp_loss,train_th))
                loss_tracking.append(temp_loss)
                          
                plt.plot(np.log10(loss_tracking))
                plt.xlabel('Epoch*%d'%int_ep)
                plt.ylabel('log10(loss)')
                file_name = os.path.join(save_folder,'ft_loss.png')
                plt.savefig(file_name)
                plt.close()
        
                
                    
                raw_data = signal_pool[:10,:,:]
                temp_dic = auto_hie(raw_data)
                
                plt.plot(temp_dic['la_l1'][1,:,2]);plt.plot(temp_dic['l1_rec'][1,:,2])
                file_name = os.path.join(save_folder,'example_final.png')
                plt.savefig(file_name)
                plt.close()
                plt.plot(temp_dic['la_l1'][1,:200,2]);plt.plot(temp_dic['l1_rec'][1,:200,2])
                file_name = os.path.join(save_folder,'example_zoomin_final.png')
                plt.savefig(file_name)
                plt.close()
            
                
                
                pickle.dump(loss_tracking, open(save_path, "wb" ) )
                
                
                loss_index_L = temp_loss
            ep += 1   
        


if __name__ == "__main__":
    
    
    
        # setup GPU
    strategy = tf.distribute.MirroredStrategy(['/gpu:0', '/gpu:1'])
    save_folder = 'training_process'
    
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
        
        
    checkpoint_path = os.path.join(save_folder,'check_point','cp.ckpt')
    checkpoint_dir = os.path.dirname(checkpoint_path)
    
    
    signal_pool, mask_pool = pickle.load( open( "signal_pool.pkl", "rb" ) )
    signal_pool = np.concatenate([signal_pool, mask_pool],-1)
    
    
    batch_size=64
    r_data = tf.data.Dataset.from_tensor_slices((signal_pool))
    r_data = r_data.shuffle(10000)
    r_data = r_data.batch(batch_size)
    r_data = r_data.prefetch(2)
    
    temp_iter = r_data.__iter__()
    x_signal = temp_iter.__next__()
    
    
    with strategy.scope():
        auto_hie = ae_hie()
        
        
        if not os.path.exists(checkpoint_dir):    
            cont_train = False
            os.mkdir(checkpoint_dir)
        else:
            print('load existing model')
            auto_hie.load_weights(checkpoint_path)
            cont_train = True


    

    
    th_seq = np.linspace(np.log10(0.05), np.log10(0.0005), num=50)
    th_seq = 10**th_seq
    lr_seq = np.linspace(np.log10(1e-3), np.log10(1e-5), num=50)
    lr_seq = 10**lr_seq


    for level in range(1,8):
        for th, lr in zip (th_seq,lr_seq):
            with strategy.scope():
                optimizer = tf.keras.optimizers.Adam(lr)
            
            loss_th = np.round(th,6)
      
            train_each_level(level=level,train_th=loss_th,int_ep=50)
            auto_hie.save_weights(checkpoint_path)
        

    

    
    for th, lr in zip (th_seq,lr_seq):
        with strategy.scope():
            ft_op = tf.keras.optimizers.Adam(lr/5)
        
        loss_th = np.round(th,6)
        ft_th = 10*loss_th
        train_each_level_ft(train_th=ft_th,int_ep=50)
        auto_hie.save_weights(checkpoint_path)