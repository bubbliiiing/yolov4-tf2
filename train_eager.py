import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping
from nets.yolo4 import yolo_body
from nets.loss import yolo_loss
import time
from utils.utils import get_random_data, get_random_data_with_Mosaic, rand, WarmUpCosineDecayScheduler, ModelCheckpoint
from functools import partial
import os

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
def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, mosaic=False):
    '''data generator for fit_generator'''
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
                    i = (i+1) % n
                else:
                    image, box = get_random_data(annotation_lines[i], input_shape)
                    i = (i+1) % n
                flag = bool(1-flag)
            else:
                image, box = get_random_data(annotation_lines[i], input_shape)
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
    # 先验框
    # 678为 142,110,  192,243,  459,401
    # 345为 36,75,  76,55,  72,146
    # 012为 12,16,  19,36,  40,28
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32') # 416,416
    # 读出xy轴，读出长宽
    # 中心点(m,n,2)
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    # 计算比例
    true_boxes[..., 0:2] = boxes_xy/input_shape[:]
    true_boxes[..., 2:4] = boxes_wh/input_shape[:]

    # m张图
    m = true_boxes.shape[0]
    # 得到网格的shape为13,13;26,26;52,52
    grid_shapes = [input_shape//{0:32, 1:16, 2:8}[l] for l in range(num_layers)]
    # y_true的格式为(m,13,13,3,85)(m,26,26,3,85)(m,52,52,3,85)
    y_true = [np.zeros((m,grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask[l]),5+num_classes),
        dtype='float32') for l in range(num_layers)]
    # [1,9,2]
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    # 长宽要大于0才有效
    valid_mask = boxes_wh[..., 0]>0

    for b in range(m):
        # 对每一张图进行处理
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh)==0: continue
        # [n,1,2]
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        # 计算真实框和哪个先验框最契合
        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)
        # 维度是(n) 感谢 消尽不死鸟 的提醒
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    # floor用于向下取整
                    i = np.floor(true_boxes[b,t,0]*grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b,t,1]*grid_shapes[l][0]).astype('int32')
                    # 找到真实框在特征层l中第b副图像对应的位置
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b,t, 4].astype('int32')
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b,t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5+c] = 1

    return y_true


@tf.function
def train_step(imgs, yolo_loss, targets, net, optimizer, regularization):
    with tf.GradientTape() as tape:
        # 计算loss
        P5_output, P4_output, P3_output = net(imgs, training=True)
        args = [P5_output, P4_output, P3_output] + targets
        loss_value = yolo_loss(args,anchors,num_classes,label_smoothing=label_smoothing)
        if regularization:
            # 加入正则化损失
            loss_value = tf.reduce_sum(net.losses) + loss_value
    grads = tape.gradient(loss_value, net.trainable_variables)
    optimizer.apply_gradients(zip(grads, net.trainable_variables))
    return loss_value

def fit_one_epoch(net, yolo_loss, optimizer, epoch, epoch_size, epoch_size_val, gen, genval, Epoch, anchors, 
                        num_classes, label_smoothing, regularization=False):
    loss = 0
    val_loss = 0
    start_time = time.time()
    for iteration, batch in enumerate(gen):
        if iteration>=epoch_size:
            break
        images, target0, target1, target2 = batch[0], batch[1], batch[2], batch[3]
        targets = [target0, target1, target2]
        targets = [tf.convert_to_tensor(target) for target in targets]
        loss_value = train_step(images, yolo_loss, targets, net, optimizer, regularization)
        loss = loss + loss_value

        waste_time = time.time() - start_time
        print('\nEpoch:'+ str(epoch+1) + '/' + str(Epoch))
        print('iter:' + str(iteration) + '/' + str(epoch_size) + ' || Total Loss: %.4f || %.4fs/step' % (loss/(iteration+1),waste_time))
        start_time = time.time()
        
    print('Start Validation')
    for iteration, batch in enumerate(genval):
        if iteration>=epoch_size_val:
            break
        # 计算验证集loss
        images, target0, target1, target2 = batch[0], batch[1], batch[2], batch[3]
        targets = [target0, target1, target2]
        targets = [tf.convert_to_tensor(target) for target in targets]

        P5_output, P4_output, P3_output = net(images)
        args = [P5_output, P4_output, P3_output] + targets
        loss_value = yolo_loss(args,anchors,num_classes,label_smoothing=label_smoothing)
        if regularization:
            # 加入正则化损失
            loss_value = tf.reduce_sum(net.losses) + loss_value
        # 更新验证集loss
        val_loss = val_loss + loss_value

    print('Finish Validation')
    print('\nEpoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (loss/(epoch_size+1),val_loss/(epoch_size_val+1)))
    net.save_weights('logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.h5'%((epoch+1),loss/(epoch_size+1),val_loss/(epoch_size_val+1)))
      
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

if __name__ == "__main__":
    #----------------------------------------------#
    #   视频中说的速度慢问题已经解决了很多
    #   现在train.py和train_eager.py速度差距不大
    #   如果还有改进速度的地方可以私信告诉我!
    #----------------------------------------------#
    # 标签的位置
    annotation_path = '2007_train.txt'
    # 获取classes和anchor的位置
    classes_path = 'model_data/voc_classes.txt'    
    anchors_path = 'model_data/yolo_anchors.txt'
    #-------------------------------------------#
    #   权值文件的下载请看README
    #   预训练模型的位置
    #-------------------------------------------#
    weights_path = 'model_data/yolo4_weight.h5'
    # 获得classes和anchor
    class_names = get_classes(classes_path)
    anchors = get_anchors(anchors_path)
    # 一共有多少类
    num_classes = len(class_names)
    num_anchors = len(anchors)
    #----------------------------------------------#
    #   输入的shape大小
    #   显存比较小可以使用416x416
    #   现存比较大可以使用608x608
    #----------------------------------------------#
    input_shape = (416,416)

    #-------------------------------#
    #   tricks的使用设置
    #-------------------------------#
    mosaic = True
    Cosine_scheduler = False
    label_smoothing = 0
    # 是否使用正则化
    regularization = True
    #-------------------------------#
    #   Dataloder的使用
    #-------------------------------#
    Use_Data_Loader = True

    # 输入的图像为
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape

    # 创建yolo模型
    print('Create YOLOv4 model with {} anchors and {} classes.'.format(num_anchors, num_classes))
    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    
    # 载入预训练权重
    print('Load weights {}.'.format(weights_path))
    model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
    
    # 0.1用于验证，0.9用于训练
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val
    
    freeze_layers = 302
    for i in range(freeze_layers): model_body.layers[i].trainable = False
    print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model_body.layers)))
    
    # 调整非主干模型first
    if True:
        Init_epoch = 0
        Freeze_epoch = 25
        # batch_size大小，每次喂入多少数据
        batch_size = 2
        # 最大学习率
        learning_rate_base = 1e-3
        if Use_Data_Loader:
            gen = partial(data_generator, annotation_lines = lines[:num_train], batch_size = batch_size, input_shape = input_shape, 
                            anchors = anchors, num_classes = num_classes, mosaic=mosaic)
            gen = tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32, tf.float32, tf.float32))
                
            gen_val = partial(data_generator, annotation_lines = lines[num_train:], batch_size = batch_size, 
                            input_shape = input_shape, anchors = anchors, num_classes = num_classes, mosaic=False)
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
                decay_rate=0.9,
                staircase=True
            )
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        for epoch in range(Init_epoch,Freeze_epoch):
            fit_one_epoch(model_body, yolo_loss, optimizer, epoch, epoch_size, epoch_size_val,gen, gen_val, 
                        Freeze_epoch, anchors, num_classes, label_smoothing, regularization)
                        
    for i in range(freeze_layers): model_body.layers[i].trainable = True

    # 解冻后训练
    if True:
        Freeze_epoch = 25
        Epoch = 50
        # batch_size大小，每次喂入多少数据
        batch_size = 2
        # 最大学习率
        learning_rate_base = 1e-4
        if Use_Data_Loader:
            gen = partial(data_generator, annotation_lines = lines[:num_train], batch_size = batch_size, input_shape = input_shape, 
                            anchors = anchors, num_classes = num_classes, mosaic=mosaic)
            gen = tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32, tf.float32, tf.float32))
                
            gen_val = partial(data_generator, annotation_lines = lines[num_train:], batch_size = batch_size, 
                            input_shape = input_shape, anchors = anchors, num_classes = num_classes, mosaic=False)
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
                decay_steps = epoch_size,
                decay_rate=0.9,
                staircase=True
            )
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        for epoch in range(Freeze_epoch,Epoch):
            fit_one_epoch(model_body, yolo_loss, optimizer, epoch, epoch_size, epoch_size_val,gen, gen_val, 
                        Epoch, anchors, num_classes, label_smoothing, regularization)
