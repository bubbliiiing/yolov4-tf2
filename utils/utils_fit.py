import os

import tensorflow as tf
from tqdm import tqdm


#------------------------------#
#   防止bug
#------------------------------#
def get_train_step_fn(strategy):
    @tf.function
    def train_step(imgs, targets, net, optimizer):
        with tf.GradientTape() as tape:
            #------------------------------#
            #   计算loss
            #------------------------------#
            loss_value = net([imgs] + targets, training=True)
            #------------------------------#
            #   添加上l2正则化参数
            #------------------------------#
            loss_value = tf.reduce_sum(net.losses) + loss_value
        grads = tape.gradient(loss_value, net.trainable_variables)
        optimizer.apply_gradients(zip(grads, net.trainable_variables))
        return loss_value
    
    if strategy == None:
        return train_step
    else:
        #----------------------#
        #   多gpu训练
        #----------------------#
        @tf.function
        def distributed_train_step(images, targets, net, optimizer):
            per_replica_losses = strategy.run(train_step, args=(images, targets, net, optimizer,))
            return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses,
                                    axis=None)
        return distributed_train_step
    
#----------------------#
#   防止bug
#----------------------#
def get_val_step_fn(strategy):
    @tf.function
    def val_step(imgs, targets, net, optimizer):
        #------------------------------#
        #   计算loss
        #------------------------------#
        loss_value = net([imgs] + targets, training=False)
        #------------------------------#
        #   添加上l2正则化参数
        #------------------------------#
        loss_value = tf.reduce_sum(net.losses) + loss_value
        return loss_value
    if strategy == None:
        return val_step
    else:
        #----------------------#
        #   多gpu验证
        #----------------------#
        @tf.function
        def distributed_val_step(images, targets, net, optimizer):
            per_replica_losses = strategy.run(val_step, args=(images, targets, net, optimizer,))
            return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses,
                                    axis=None)
        return distributed_val_step

def fit_one_epoch(net, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, save_period, save_dir, strategy):
    train_step  = get_train_step_fn(strategy)
    val_step    = get_val_step_fn(strategy)
    
    loss        = 0
    val_loss    = 0
    print('Start Train')
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break
            images, target0, target1, target2 = batch[0], batch[1], batch[2], batch[3]
            targets     = [target0, target1, target2]
            loss_value  = train_step(images, targets, net, optimizer)
            loss        = loss + loss_value

            pbar.set_postfix(**{'total_loss': float(loss) / (iteration + 1), 
                                'lr'        : optimizer.lr.numpy()})
            pbar.update(1)
    print('Finish Train')
            
    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            images, target0, target1, target2 = batch[0], batch[1], batch[2], batch[3]
            targets     = [target0, target1, target2]
            loss_value  = val_step(images, targets, net, optimizer)
            val_loss    = val_loss + loss_value

            pbar.set_postfix(**{'total_loss': float(val_loss) / (iteration + 1)})
            pbar.update(1)
    print('Finish Validation')

    logs = {'loss': loss.numpy() / epoch_step, 'val_loss': val_loss.numpy() / epoch_step_val}
    loss_history.on_epoch_end([], logs)
    eval_callback.on_epoch_end(epoch, logs)
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))
    
    #-----------------------------------------------#
    #   保存权值
    #-----------------------------------------------#
    if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
        net.save_weights(os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.h5" % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))
        
    if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
        print('Save best model to best_epoch_weights.pth')
        net.save_weights(os.path.join(save_dir, "best_epoch_weights.h5"))
            
    net.save_weights(os.path.join(save_dir, "last_epoch_weights.h5"))
