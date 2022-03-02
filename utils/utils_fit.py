import tensorflow as tf
from nets.yolo_training import yolo_loss
from tqdm import tqdm


#------------------------------#
#   防止bug
#------------------------------#
def get_train_step_fn(input_shape, anchors, anchors_mask, num_classes, label_smoothing):
    @tf.function
    def train_step(imgs, targets, net, optimizer):
        with tf.GradientTape() as tape:
            #------------------------------#
            #   计算loss
            #------------------------------#
            P5_output, P4_output, P3_output = net(imgs, training=True)
            args        = [P5_output, P4_output, P3_output] + targets
            loss_value  = yolo_loss(
                args, input_shape, anchors, anchors_mask, num_classes, 
                label_smoothing = label_smoothing,
                balance         = [0.4, 1.0, 4],
                box_ratio       = 0.05, 
                obj_ratio       = 5 * (input_shape[0] * input_shape[1]) / (416 ** 2),
                cls_ratio       = 1 * (num_classes / 80)
            )
            #------------------------------#
            #   添加上l2正则化参数
            #------------------------------#
            loss_value  = tf.reduce_sum(net.losses) + loss_value
        grads = tape.gradient(loss_value, net.trainable_variables)
        optimizer.apply_gradients(zip(grads, net.trainable_variables))
        return loss_value
    return train_step

def fit_one_epoch(net, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, 
            input_shape, anchors, anchors_mask, num_classes, label_smoothing, save_period):
    train_step  = get_train_step_fn(input_shape, anchors, anchors_mask, num_classes, label_smoothing)
    loss        = 0
    val_loss    = 0
    print('Start Train')
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break
            images, target0, target1, target2 = batch[0], batch[1], batch[2], batch[3]
            targets     = [target0, target1, target2]
            targets     = [tf.convert_to_tensor(target) for target in targets]
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
            targets     = [tf.convert_to_tensor(target) for target in targets]
            #------------------------------#
            #   计算loss
            #------------------------------#
            P5_output, P4_output, P3_output = net(images)
            args        = [P5_output, P4_output, P3_output] + targets
            loss_value  = yolo_loss(args, input_shape, anchors, anchors_mask, num_classes, label_smoothing=label_smoothing)
            #------------------------------#
            #   添加上l2正则化参数
            #------------------------------#
            loss_value  = tf.reduce_sum(net.losses) + loss_value
            val_loss = val_loss + loss_value

            pbar.set_postfix(**{'total_loss': float(val_loss) / (iteration + 1)})
            pbar.update(1)
    print('Finish Validation')

    logs = {'loss': loss.numpy() / epoch_step, 'val_loss': val_loss.numpy() / epoch_step_val}
    loss_history.on_epoch_end([], logs)
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))
    if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
        net.save_weights('logs/ep%03d-loss%.3f-val_loss%.3f.h5' % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val))
