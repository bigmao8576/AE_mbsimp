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
    
    dic = auto_hie(x)
    diff_loss = dis_loss_pair(dic['la_l%s'%level],dic['l%s_int'%level])
    
    return diff_loss


def train_step_L(level):
    @tf.function
    def distributed_apply_grads_L(dataset_inputs):
        per_replica_losses = strategy.run(apply_grads_L, args=(dataset_inputs,level))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                                 axis=None)*0.5
    return distributed_apply_grads_L









def apply_grads_ft(x,level):
    if level == 2:

            
        with tf.GradientTape() as tape:
            dic = auto_hie(x)
            temp_l1 = auto_hie.cde_L1(dic['l2_int'])
            
            diff_loss = dis_loss_pair(x,temp_l1)+dis_loss_pair(dic['la_l2'],dic['l2_int'])
        
        name_ls = ['cen_l1','cde_l1','cen_l2','cde_l2']
        var = [item for item in auto_hie.trainable_variables if any(st in name_ls for st in item.name.split('/'))]
        grad= tape.gradient(diff_loss, var)
        ft_op.apply_gradients(zip(grad, var)) 
        
        dic = auto_hie(x)
        temp_l1 = auto_hie.cde_L1(dic['l2_int'])
        diff_loss = dis_loss_pair(x,temp_l1)
    
        return diff_loss                
                
    elif level ==3:

        with tf.GradientTape() as tape:
            dic = auto_hie(x)
            temp_l2 = auto_hie.cde_L2(dic['l3_int'])
            temp_l1 = auto_hie.cde_L1(temp_l2)
            
            diff_loss =(dis_loss_pair(x,temp_l1) +
                        dis_loss_pair(dic['la_l2'],temp_l2) +
                        dis_loss_pair(dic['la_l3'],dic['l3_int']))
    
        name_ls = ['cen_l1','cde_l1','cen_l2','cde_l2','cen_l3','cde_l3']
        var = [item for item in auto_hie.trainable_variables if any(st in name_ls for st in item.name.split('/'))]
        grad= tape.gradient(diff_loss, var)
        ft_op.apply_gradients(zip(grad, var)) 
        
        dic = auto_hie(x)
        temp_l2 = auto_hie.cde_L2(dic['l3_int'])
        temp_l1 = auto_hie.cde_L1(temp_l2)
        diff_loss = dis_loss_pair(x,temp_l1)
    
        return diff_loss 
            
    elif level ==4:

        with tf.GradientTape() as tape:
            dic = auto_hie(x)
            temp_l3 = auto_hie.cde_L3(dic['l4_int'])
            temp_l2 = auto_hie.cde_L2(temp_l3)
            temp_l1 = auto_hie.cde_L1(temp_l2)
            
            diff_loss =(dis_loss_pair(x,temp_l1) +
                        dis_loss_pair(dic['la_l2'],temp_l2) +
                        dis_loss_pair(dic['la_l3'],temp_l3) +
                        dis_loss_pair(dic['la_l4'],dic['l4_int']))
    
        name_ls = ['cen_l1','cde_l1','cen_l2','cde_l2','cen_l3','cde_l3','cen_l4','cde_l4']
        var = [item for item in auto_hie.trainable_variables if any(st in name_ls for st in item.name.split('/'))]
        grad= tape.gradient(diff_loss, var)
        ft_op.apply_gradients(zip(grad, var)) 
        
        dic = auto_hie(x)
        temp_l3 = auto_hie.cde_L3(dic['l4_int'])
        temp_l2 = auto_hie.cde_L2(temp_l3)
        temp_l1 = auto_hie.cde_L1(temp_l2)
        diff_loss = dis_loss_pair(x,temp_l1)
    
        return diff_loss 
            
    elif level ==5:

        with tf.GradientTape() as tape:
            dic = auto_hie(x)
            temp_l4 = auto_hie.cde_L4(dic['l5_int'])
            temp_l3 = auto_hie.cde_L3(temp_l4)
            temp_l2 = auto_hie.cde_L2(temp_l3)
            temp_l1 = auto_hie.cde_L1(temp_l2)
            
            diff_loss =(dis_loss_pair(x,temp_l1) +
                        dis_loss_pair(dic['la_l2'],temp_l2) +
                        dis_loss_pair(dic['la_l3'],temp_l3) +
                        dis_loss_pair(dic['la_l4'],temp_l4) +
                        dis_loss_pair(dic['la_l5'],dic['l5_int']))
    
        name_ls = ['cen_l1','cde_l1','cen_l2','cde_l2','cen_l3','cde_l3','cen_l4','cde_l4','cen_l5','cde_l5']
        var = [item for item in auto_hie.trainable_variables if any(st in name_ls for st in item.name.split('/'))]
        grad= tape.gradient(diff_loss, var)
        ft_op.apply_gradients(zip(grad, var)) 
        
        dic = auto_hie(x)
        temp_l4 = auto_hie.cde_L4(dic['l5_int'])
        temp_l3 = auto_hie.cde_L3(temp_l4)
        temp_l2 = auto_hie.cde_L2(temp_l3)
        temp_l1 = auto_hie.cde_L1(temp_l2)
        diff_loss = dis_loss_pair(x,temp_l1)
    
        return diff_loss 
            
    elif level ==6:

        with tf.GradientTape() as tape:
            dic = auto_hie(x)
            temp_l5 = auto_hie.cde_L5(dic['l6_int'])
            temp_l4 = auto_hie.cde_L4(temp_l5)
            temp_l3 = auto_hie.cde_L3(temp_l4)
            temp_l2 = auto_hie.cde_L2(temp_l3)
            temp_l1 = auto_hie.cde_L1(temp_l2)
            
            diff_loss =(dis_loss_pair(x,temp_l1) +
                        dis_loss_pair(dic['la_l2'],temp_l2) +
                        dis_loss_pair(dic['la_l3'],temp_l3) +
                        dis_loss_pair(dic['la_l4'],temp_l4) +
                        dis_loss_pair(dic['la_l5'],temp_l5) +
                        dis_loss_pair(dic['la_l6'],dic['l6_int']))
    
        name_ls = ['cen_l1','cde_l1','cen_l2','cde_l2','cen_l3','cde_l3','cen_l4','cde_l4','cen_l5','cde_l5','cen_l6','cde_l6']
        var = [item for item in auto_hie.trainable_variables if any(st in name_ls for st in item.name.split('/'))]
        grad= tape.gradient(diff_loss, var)
        ft_op.apply_gradients(zip(grad, var)) 
        
        dic = auto_hie(x)
        temp_l5 = auto_hie.cde_L5(dic['l6_int'])
        temp_l4 = auto_hie.cde_L4(temp_l5)
        temp_l3 = auto_hie.cde_L3(temp_l4)
        temp_l2 = auto_hie.cde_L2(temp_l3)
        temp_l1 = auto_hie.cde_L1(temp_l2)
        diff_loss = dis_loss_pair(x,temp_l1)
    
        return diff_loss 
            
    elif level ==7:

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
        dic = auto_hie(x)
        diff_loss = dis_loss_pair(x,dic['l1_rec'])
    
        return diff_loss 


def train_step_ft(level):
    @tf.function
    def distributed_apply_grads_ft(dataset_inputs):
        per_replica_losses = strategy.run(apply_grads_ft, args=(dataset_inputs,level))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                                 axis=None)*0.5
    return distributed_apply_grads_ft







def train_each_level(level,train_th,int_ep):
    with strategy.scope():
        save_path = os.path.join(save_folder,'L%d_loss.pkl'%level)
        loss_index_L = 10.0
        if os.path.exists(save_path):
            loss_tracking = pickle.load(open(save_path, "rb" ))
            ep = (len(loss_tracking)-1)*10+1

        else:
            loss_tracking = []
            ep = 0

            
        train_L = train_step_L(level=level)
            
        while loss_index_L > train_th:
            ep_loss = []
        
            
            for x_data in r_data: 
                diff_loss = train_L(x_data)    
                ep_loss.append(diff_loss.numpy())
                
            if ep%int_ep==0:   
                print(ep,'_L%d____'%level,np.mean(ep_loss),'---',train_th)
                loss_tracking.append(np.mean(ep_loss))
        
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
                
                
                loss_index_L = np.mean(ep_loss)
            ep += 1   
        return loss_tracking



def train_each_level_ft(level,train_th,int_ep):
    loss_index_L = 10.0
    save_path = os.path.join(save_folder,'L%d_ft_loss.pkl'%level)
    if os.path.exists(save_path):
        loss_tracking = pickle.load(open(save_path, "rb" ))
        ep = (len(loss_tracking)-1)*10+1

    else:
        loss_tracking = []
        ep = 0

        
    train_ft = train_step_ft(level=level)
        
    while loss_index_L > train_th:
        ep_loss = []
    
        
        for x_data in r_data: 
            diff_loss = train_ft(x_data)    
            ep_loss.append(diff_loss.numpy())
            
        if ep%int_ep==0:   
            print(ep,'_L%d_ft___'%level,np.mean(ep_loss),'---',train_th)
            loss_tracking.append(np.mean(ep_loss))
                      
            plt.plot(np.log10(loss_tracking))
            plt.xlabel('Epoch*%d'%int_ep)
            plt.ylabel('log10(loss)')
            file_name = os.path.join(save_folder,'L%d_ft_loss.png'%level)
            plt.savefig(file_name)
            plt.close()
    
            if level == 7:
                
                raw_data = signal_pool[:10,:,:]
                temp_dic = auto_hie(raw_data)
                
                plt.plot(temp_dic['la_l1'][1,:,2]);plt.plot(temp_dic['l1_rec'][1,:,2])
                file_name = os.path.join(save_folder,'example_final.png')
                plt.savefig(file_name)
                plt.close()
                plt.plot(temp_dic['la_l1'][1,:200,2]);plt.plot(temp_dic['la_l1'][1,:200,2])
                file_name = os.path.join(save_folder,'example_zoomin_final.png')
                plt.savefig(file_name)
                plt.close()
            
            
            
            pickle.dump(loss_tracking, open(save_path, "wb" ) )
            
            
            loss_index_L = np.mean(ep_loss)
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


    

    
    th_seq = np.linspace(np.log10(0.05), -4, num=100)
    th_seq = 10**th_seq
    lr_seq = np.linspace(np.log10(2e-5), -7, num=100)
    lr_seq = 10**lr_seq
    
    for th, lr in zip (th_seq,lr_seq):
        
        
        
        with strategy.scope():
            optimizer = tf.keras.optimizers.RMSprop(lr)
            ft_op = tf.keras.optimizers.RMSprop(lr/10)
        
        loss_th = np.round(th,6)
        ft_th = 5*loss_th
    

    
        total_loss = train_each_level(level=1,train_th=loss_th,int_ep=10)
        auto_hie.save_weights(checkpoint_path)
        
        total_loss = train_each_level(level=2,train_th=loss_th,int_ep=10)
        auto_hie.save_weights(checkpoint_path)
        
        total_loss = train_each_level_ft(level=2,train_th=ft_th,int_ep=10)
        auto_hie.save_weights(checkpoint_path)
        
        total_loss = train_each_level(level=3,train_th=loss_th,int_ep=10)
        auto_hie.save_weights(checkpoint_path)
        
        total_loss = train_each_level_ft(level=3,train_th=ft_th,int_ep=10)
        auto_hie.save_weights(checkpoint_path)
        
        total_loss = train_each_level(level=4,train_th=loss_th,int_ep=10)
        auto_hie.save_weights(checkpoint_path)
        
        total_loss = train_each_level_ft(level=4,train_th=ft_th,int_ep=10)
        auto_hie.save_weights(checkpoint_path)
        
        total_loss = train_each_level(level=5,train_th=loss_th,int_ep=10)
        auto_hie.save_weights(checkpoint_path)
        
        total_loss = train_each_level_ft(level=5,train_th=ft_th,int_ep=10)
        auto_hie.save_weights(checkpoint_path)
        
        total_loss = train_each_level(level=6,train_th=loss_th,int_ep=10)
        auto_hie.save_weights(checkpoint_path)
        
        total_loss = train_each_level_ft(level=6,train_th=ft_th,int_ep=10)
        auto_hie.save_weights(checkpoint_path)
        
        total_loss = train_each_level(level=7,train_th=loss_th,int_ep=10)
        auto_hie.save_weights(checkpoint_path)
        
        total_loss = train_each_level_ft(level=7,train_th=ft_th,int_ep=10)
        auto_hie.save_weights(checkpoint_path)
    
