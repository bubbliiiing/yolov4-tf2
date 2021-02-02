import os
import time
from functools import partial

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                        TensorBoard)
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

from nets.loss import yolo_loss
from nets.yolo4 import yolo_body
from utils.utils import (ModelCheckpoint, WarmUpCosineDecayScheduler,
                         get_random_data, get_random_data_with_Mosaic, rand)


#---------------------------------------------------#
#   获得类和先验框
#---------------------------------------------------#
def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

#---------------------------------------------------#
#   训练数据生成器
#---------------------------------------------------#
def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, mosaic=False, random=True):
    n = len(annotation_lines)
    i = 0
    flag = True
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            if mosaic:
                if flag and (i+4) < n:
                    image, box = get_random_data_with_Mosaic(annotation_lines[i:i+4], input_shape)
                    i = (i+4) % n
                else:
                    image, box = get_random_data(annotation_lines[i], input_shape, random=random)
                    i = (i+1) % n
                flag = bool(1-flag)
            else:
                image, box = get_random_data(annotation_lines[i], input_shape, random=random)
                i = (i+1) % n
            image_data.append(image)
            box_data.append(box)
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield image_data, y_true[0], y_true[1], y_true[2]

#---------------------------------------------------#
#   读入xml文件，并输出y_true
#---------------------------------------------------#
def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    assert (true_boxes[..., 4]<num_classes).all(), 'class id must be less than num_classes'
    # 一共有三个特征层数
    num_layers = len(anchors)//3
    #-----------------------------------------------------------#
    #   13x13的特征层对应的anchor是[142, 110], [192, 243], [459, 401]
    #   26x26的特征层对应的anchor是[36, 75], [76, 55], [72, 146]
    #   52x52的特征层对应的anchor是[12, 16], [19, 36], [40, 28]
    #-----------------------------------------------------------#
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]]

    #-----------------------------------------------------------#
    #   获得框的坐标和图片的大小
    #-----------------------------------------------------------#
    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    #-----------------------------------------------------------#
    #   通过计算获得真实框的中心和宽高
    #   中心点(m,n,2) 宽高(m,n,2)
    #-----------------------------------------------------------#
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    #-----------------------------------------------------------#
    #   将真实框归一化到小数形式
    #-----------------------------------------------------------#
    true_boxes[..., 0:2] = boxes_xy/input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh/input_shape[::-1]

    # m为图片数量，grid_shapes为网格的shape
    m = true_boxes.shape[0]
    grid_shapes = [input_shape//{0:32, 1:16, 2:8}[l] for l in range(num_layers)]
    #-----------------------------------------------------------#
    #   y_true的格式为(m,13,13,3,85)(m,26,26,3,85)(m,52,52,3,85)
    #-----------------------------------------------------------#
    y_true = [np.zeros((m,grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask[l]),5+num_classes),
        dtype='float32') for l in range(num_layers)]

    #-----------------------------------------------------------#
    #   [9,2] -> [1,9,2]
    #-----------------------------------------------------------#
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes

    #-----------------------------------------------------------#
    #   长宽要大于0才有效
    #-----------------------------------------------------------#
    valid_mask = boxes_wh[..., 0]>0

    for b in range(m):
        # 对每一张图进行处理
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh)==0: continue
        #-----------------------------------------------------------#
        #   [n,2] -> [n,1,2]
        #-----------------------------------------------------------#
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        #-----------------------------------------------------------#
        #   计算所有真实框和先验框的交并比
        #   intersect_area  [n,9]
        #   box_area        [n,1]
        #   anchor_area     [1,9]
        #   iou             [n,9]
        #-----------------------------------------------------------#
        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]

        iou = intersect_area / (box_area + anchor_area - intersect_area)
        #-----------------------------------------------------------#
        #   维度是[n,] 感谢 消尽不死鸟 的提醒
        #-----------------------------------------------------------#
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            #-----------------------------------------------------------#
            #   找到每个真实框所属的特征层
            #-----------------------------------------------------------#
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    #-----------------------------------------------------------#
                    #   floor用于向下取整，找到真实框所属的特征层对应的x、y轴坐标
                    #-----------------------------------------------------------#
                    i = np.floor(true_boxes[b,t,0] * grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b,t,1] * grid_shapes[l][0]).astype('int32')
                    #-----------------------------------------------------------#
                    #   k指的的当前这个特征点的第k个先验框
                    #-----------------------------------------------------------#
                    k = anchor_mask[l].index(n)
                    #-----------------------------------------------------------#
                    #   c指的是当前这个真实框的种类
                    #-----------------------------------------------------------#
                    c = true_boxes[b, t, 4].astype('int32')
                    #-----------------------------------------------------------#
                    #   y_true的shape为(m,13,13,3,85)(m,26,26,3,85)(m,52,52,3,85)
                    #   最后的85可以拆分成4+1+80，4代表的是框的中心与宽高、
                    #   1代表的是置信度、80代表的是种类
                    #-----------------------------------------------------------#
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5+c] = 1

    return y_true
# 防止bug
def get_train_step_fn():
    @tf.function
    def train_step(imgs, yolo_loss, targets, net, optimizer, regularization, normalize):
        with tf.GradientTape() as tape:
            # 计算loss
            P5_output, P4_output, P3_output = net(imgs, training=True)
            args = [P5_output, P4_output, P3_output] + targets
            loss_value = yolo_loss(args,anchors,num_classes,label_smoothing=label_smoothing,normalize=normalize)
            if regularization:
                # 加入正则化损失
                loss_value = tf.reduce_sum(net.losses) + loss_value
        grads = tape.gradient(loss_value, net.trainable_variables)
        optimizer.apply_gradients(zip(grads, net.trainable_variables))
        return loss_value
    return train_step

def fit_one_epoch(net, yolo_loss, optimizer, epoch, epoch_size, epoch_size_val, gen, genval, Epoch, anchors, 
                        num_classes, label_smoothing, regularization=False, normalize=True, train_step=None):
    loss = 0
    val_loss = 0
    start_time = time.time()
    with tqdm(total=epoch_size,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration>=epoch_size:
                break
            images, target0, target1, target2 = batch[0], batch[1], batch[2], batch[3]
            targets = [target0, target1, target2]
            targets = [tf.convert_to_tensor(target) for target in targets]
            loss_value = train_step(images, yolo_loss, targets, net, optimizer, regularization, normalize)
            loss = loss + loss_value.numpy()

            waste_time = time.time() - start_time
            pbar.set_postfix(**{'total_loss': float(loss) / (iteration + 1), 
                                'lr'        : optimizer._decayed_lr(tf.float32).numpy(),
                                'step/s'    : waste_time})
            pbar.update(1)
            start_time = time.time()
            
    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration>=epoch_size_val:
                break
            # 计算验证集loss
            images, target0, target1, target2 = batch[0], batch[1], batch[2], batch[3]
            targets = [target0, target1, target2]
            targets = [tf.convert_to_tensor(target) for target in targets]

            P5_output, P4_output, P3_output = net(images)
            args = [P5_output, P4_output, P3_output] + targets
            loss_value = yolo_loss(args,anchors,num_classes,label_smoothing=label_smoothing, normalize=normalize)
            if regularization:
                # 加入正则化损失
                loss_value = tf.reduce_sum(net.losses) + loss_value
            # 更新验证集loss
            val_loss = val_loss + loss_value.numpy()

            pbar.set_postfix(**{'total_loss': float(val_loss)/ (iteration + 1)})
            pbar.update(1)

    print('Finish Validation')
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (loss/(epoch_size+1),val_loss/(epoch_size_val+1)))
    net.save_weights('logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.h5'%((epoch+1),loss/(epoch_size+1),val_loss/(epoch_size_val+1)))
      
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

#----------------------------------------------------#
#   检测精度mAP和pr曲线计算参考视频
#   https://www.bilibili.com/video/BV1zE411u7Vw
#----------------------------------------------------#
if __name__ == "__main__":
    #----------------------------------------------#
    #   视频中说的速度慢问题已经解决了很多
    #   现在train.py和train_eager.py速度差距不大
    #   如果还有改进速度的地方可以私信告诉我!
    #----------------------------------------------#
    #----------------------------------------------------#
    #   获得图片路径和标签
    #----------------------------------------------------#
    annotation_path = '2007_train.txt'
    #------------------------------------------------------#
    #   训练后的模型保存的位置，保存在logs文件夹里面
    #------------------------------------------------------#
    log_dir = 'logs/'
    #----------------------------------------------------#
    #   classes和anchor的路径，非常重要
    #   训练前一定要修改classes_path，使其对应自己的数据集
    #----------------------------------------------------#
    classes_path = 'model_data/voc_classes.txt'    
    anchors_path = 'model_data/yolo_anchors.txt'
    #------------------------------------------------------#
    #   权值文件请看README，百度网盘下载
    #   训练自己的数据集时提示维度不匹配正常
    #   预测的东西都不一样了自然维度不匹配
    #------------------------------------------------------#
    weights_path = 'model_data/yolo4_weight.h5'
    #------------------------------------------------------#
    #   训练用图片大小
    #   一般在416x416和608x608选择
    #------------------------------------------------------#
    input_shape = (416,416)
    #------------------------------------------------------#
    #   是否对损失进行归一化，用于改变loss的大小
    #   用于决定计算最终loss是除上batch_size还是除上正样本数量
    #------------------------------------------------------#
    normalize = False
    #----------------------------------------------------#
    #   获取classes和anchor
    #----------------------------------------------------#
    class_names = get_classes(classes_path)
    anchors = get_anchors(anchors_path)
    #------------------------------------------------------#
    #   一共有多少类和多少先验框
    #------------------------------------------------------#
    num_classes = len(class_names)
    num_anchors = len(anchors)
    #------------------------------------------------------#
    #   Yolov4的tricks应用
    #   mosaic 马赛克数据增强 True or False 
    #   实际测试时mosaic数据增强并不稳定，所以默认为False
    #   Cosine_scheduler 余弦退火学习率 True or False
    #   label_smoothing 标签平滑 0.01以下一般 如0.01、0.005
    #------------------------------------------------------#
    mosaic = False
    Cosine_scheduler = False
    label_smoothing = 0

    regularization = True
    #-------------------------------#
    #   Dataloder的使用
    #-------------------------------#
    Use_Data_Loader = True

    #------------------------------------------------------#
    #   创建yolo模型
    #------------------------------------------------------#
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    print('Create YOLOv4 model with {} anchors and {} classes.'.format(num_anchors, num_classes))
    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    
    #------------------------------------------------------#
    #   载入预训练权重
    #------------------------------------------------------#
    print('Load weights {}.'.format(weights_path))
    model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
    
    #----------------------------------------------------------------------#
    #   验证集的划分在train.py代码里面进行
    #   2007_test.txt和2007_val.txt里面没有内容是正常的。训练不会使用到。
    #   当前划分方式下，验证集和训练集的比例为1:9
    #----------------------------------------------------------------------#
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val
    
    #------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    #------------------------------------------------------#
    freeze_layers = 249
    for i in range(freeze_layers): model_body.layers[i].trainable = False
    print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model_body.layers)))
    
    # 调整非主干模型first
    if True:
        Init_epoch = 0
        Freeze_epoch = 50
        batch_size = 2
        learning_rate_base = 1e-3

        if Use_Data_Loader:
            gen = partial(data_generator, annotation_lines = lines[:num_train], batch_size = batch_size, input_shape = input_shape, 
                            anchors = anchors, num_classes = num_classes, mosaic=mosaic, random=True)
            gen = tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32, tf.float32, tf.float32))
                
            gen_val = partial(data_generator, annotation_lines = lines[num_train:], batch_size = batch_size, 
                            input_shape = input_shape, anchors = anchors, num_classes = num_classes, mosaic=False, random=False)
            gen_val = tf.data.Dataset.from_generator(gen_val, (tf.float32, tf.float32, tf.float32, tf.float32))

            gen = gen.shuffle(buffer_size=batch_size).prefetch(buffer_size=batch_size)
            gen_val = gen_val.shuffle(buffer_size=batch_size).prefetch(buffer_size=batch_size)

        else:
            gen = data_generator(lines[:num_train], batch_size, input_shape, anchors, num_classes, mosaic=mosaic)
            gen_val = data_generator(lines[num_train:], batch_size, input_shape, anchors, num_classes, mosaic=False)
            
        epoch_size = num_train//batch_size
        epoch_size_val = num_val//batch_size

        if Cosine_scheduler:
            lr_schedule = tf.keras.experimental.CosineDecayRestarts(
                initial_learning_rate = learning_rate_base, 
                first_decay_steps = 5*epoch_size, 
                t_mul = 1.0,
                alpha = 1e-2
            )
        else:
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=learning_rate_base,
                decay_steps=epoch_size,
                decay_rate=0.92,
                staircase=True
            )
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        for epoch in range(Init_epoch,Freeze_epoch):
            fit_one_epoch(model_body, yolo_loss, optimizer, epoch, epoch_size, epoch_size_val,gen, gen_val, 
                        Freeze_epoch, anchors, num_classes, label_smoothing, regularization, normalize, get_train_step_fn())
                        
    for i in range(freeze_layers): model_body.layers[i].trainable = True

    # 解冻后训练
    if True:
        Freeze_epoch = 50
        Epoch = 100
        batch_size = 2
        learning_rate_base = 1e-4

        if Use_Data_Loader:
            gen = partial(data_generator, annotation_lines = lines[:num_train], batch_size = batch_size, input_shape = input_shape, 
                            anchors = anchors, num_classes = num_classes, mosaic=mosaic, random=True)
            gen = tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32, tf.float32, tf.float32))
                
            gen_val = partial(data_generator, annotation_lines = lines[num_train:], batch_size = batch_size, 
                            input_shape = input_shape, anchors = anchors, num_classes = num_classes, mosaic=False, random=False)
            gen_val = tf.data.Dataset.from_generator(gen_val, (tf.float32, tf.float32, tf.float32, tf.float32))

            gen = gen.shuffle(buffer_size=batch_size).prefetch(buffer_size=batch_size)
            gen_val = gen_val.shuffle(buffer_size=batch_size).prefetch(buffer_size=batch_size)
        else:
            gen = data_generator(lines[:num_train], batch_size, input_shape, anchors, num_classes, mosaic=mosaic, random=True),
            gen_val = data_generator(lines[num_train:], batch_size, input_shape, anchors, num_classes, mosaic=False, random=False)
            
        epoch_size = num_train//batch_size
        epoch_size_val = num_val//batch_size
        if Cosine_scheduler:
            lr_schedule = tf.keras.experimental.CosineDecayRestarts(
                initial_learning_rate = learning_rate_base, 
                first_decay_steps = 5*epoch_size, 
                t_mul = 1.0,
                alpha = 1e-2
            )
        else:
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=learning_rate_base,
                decay_steps = epoch_size,
                decay_rate=0.92,
                staircase=True
            )
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        for epoch in range(Freeze_epoch,Epoch):
            fit_one_epoch(model_body, yolo_loss, optimizer, epoch, epoch_size, epoch_size_val,gen, gen_val, 
                        Epoch, anchors, num_classes, label_smoothing, regularization, normalize, get_train_step_fn())
