import tensorflow as tf
# 通过导入模块，使用抽象keras backend API写出兼容theano和tensorflow两种backend的代码
# theano/tensorflow：作为后端（backend）引擎为keras模块提供服务
from keras import backend as K


#---------------------------------------------------#
#   对box进行调整，使其符合真实图片的样子
#---------------------------------------------------#
def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image):
    #-----------------------------------------------------------------#
    #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
    #-----------------------------------------------------------------#
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    #  K.cast --> 将张量转换到不同的 dtype 并返回。
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))

    if letterbox_image:
        #-----------------------------------------------------------------#
        #   这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
        #   new_shape指的是宽高缩放情况
        #-----------------------------------------------------------------#
        new_shape = K.round(image_shape * K.min(input_shape/image_shape))
        offset  = (input_shape - new_shape)/2./input_shape
        scale   = input_shape/new_shape

        box_yx  = (box_yx - offset) * scale
        box_hw *= scale

    box_mins    = box_yx - (box_hw / 2.)
    box_maxes   = box_yx + (box_hw / 2.)
    boxes  = K.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]])
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes

#---------------------------------------------------#
#   将预测值的每个特征层调成真实值
#---------------------------------------------------#
def get_anchors_and_decode(feats, anchors, num_classes, input_shape, calc_loss=False):
    '''
        以特征层是 13*13 为例
        :param  feats:某特征层的预测结果 形状：[batch_size,13,13,3*(5+num_class)]
                                    其中：5=4+1,4：先验框调整参数 1：判断先验框内部是否包含物体
                anchors:每一个特征层的每一个特征点的三个先验框 形状：[3,2]
                num_classes:数据集种类个数 以coco数据集为例：80

    '''
    num_anchors = len(anchors) #3
    #------------------------------------------#
    #   grid_shape指：获得特征层的高和宽
    #   以13*13特征层为例：grid_shape [13,13]
    #------------------------------------------#
    grid_shape = K.shape(feats)[1:3]
    #--------------------------------------------------------------------#
    #   获得各个特征点的坐标信息。生成的shape为(13, 13, num_anchors, 2)
    ##  以此获得每一个特征点在X轴和Y轴上面的坐标信息
    #   K.tile(x, n) --> 创建一个用 n 平铺 的 x 张量
    #--------------------------------------------------------------------#
    ## 1 K.arange(0, stop=grid_shape[1])：获得特征点的高 遍历后的结果 ->如果是13*13 相当于获取了一个（0,12）的矩阵
    ## 2 K.reshape(K.arange(...)),对获取到的矩阵样式进行调整
    ## 3.K.tile:对1,2步骤进行重复，重复次数为：[grid_shape[0], 1, num_anchors, 1]
    ###   [w] -> arrange:[13] -> reshape:[1,13,1,1]-> tile:[13,13,3,1]
    grid_x  = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]), [grid_shape[0], 1, num_anchors, 1])
    ## 1.K.arange(0, stop=grid_shape[0])：获得特征点的宽 遍历后的结果（同上）
    ## 2.K.reshape(...) 同上
    ##   [h] -> arrange:[13] -> reshape:[13,1,1,1]->tile:[13,13,3,1]
    grid_y  = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[1], num_anchors, 1])
    ## 对获取到的X轴和Y轴坐标信息进行堆叠 -- 获得一个[13,13,3,2]的矩阵，是每一个特征点的坐标信息
    grid    = K.cast(K.concatenate([grid_x, grid_y]), K.dtype(feats))
    #---------------------------------------------------------------#
    #   将先验框进行拓展，生成shape为(13, 13, num_anchors, 2)
    #---------------------------------------------------------------#
    # 经过reshape()后先验框的大小为：[1,1,3,2]
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, num_anchors, 2])
    # 经过tile()后先验框的大小为：[13,13,3,2]
    anchors_tensor = K.tile(anchors_tensor, [grid_shape[0], grid_shape[1], 1, 1])


    #---------------------------------------------------#
    #   将预测结果调整成(batch_size,13,13,3,85)
    #   85可拆分成4 + 1 + 80
    #   4代表的是中心宽高的调整参数
    #   1代表的是框的置信度
    #   80代表的是种类的置信度
    #---------------------------------------------------#
    ## 对预测结果进行reshape,对原始结果的最后一维进行拆分
        ## 原始预测结果：[batch_size,13,13,3*(5+num_class)] --> reshape:[batch_size,13,13,3,5+num_class]
        ## feats:[batch_size,h,w,anchors(先验框数量)，先验框调整参数+框内部是否包含物体+物体种类]
    feats           = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])
    #------------------------------------------#
    #   对先验框进行解码，并进行归一化
    #------------------------------------------#
    ## 取出前两个维度，利用sigmoid对先验框的中心进行调整
    box_xy          = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    ## 对宽高进行调整  anchors_tensor：先验框的宽高
    box_wh          = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    #------------------------------------------#
    #   获得预测框的置信度
    #------------------------------------------#
    box_confidence  = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])
    
    #---------------------------------------------------------------------#
    #   在计算loss的时候返回grid, feats, box_xy, box_wh
    #   在预测的时候返回box_xy, box_wh, box_confidence, box_class_probs
    #---------------------------------------------------------------------#
    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs

#---------------------------------------------------#
#   图片预测
#---------------------------------------------------#
def DecodeBox(outputs,
            anchors,
            num_classes,
            image_shape,
            input_shape,
            #-----------------------------------------------------------#
            #   13x13的特征层对应的anchor是[116,90],[156,198],[373,326]
            #   26x26的特征层对应的anchor是[30,61],[62,45],[59,119]
            #   52x52的特征层对应的anchor是[10,13],[16,30],[33,23]
            #-----------------------------------------------------------#
            anchor_mask     = [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
            max_boxes       = 100,
            confidence      = 0.5,
            nms_iou         = 0.3,
            letterbox_image = True):
    '''
        对图像预测框进行解码，预测，非极大值抑制（NMS）
        return：将NMS后保留的框的：坐标、得分、种类全部返回
    '''
    box_xy = []
    box_wh = []
    box_confidence  = []
    box_class_probs = []
    ## 对每一个特征层进行循环，将每一个特征层的先验框传入get_anchors_and_decode（）函数中，进行先验框的解码与生成
    for i in range(len(outputs)):
        ## 获取到先验框的解码预测结果
        sub_box_xy, sub_box_wh, sub_box_confidence, sub_box_class_probs = \
            get_anchors_and_decode(outputs[i], anchors[anchor_mask[i]], num_classes, input_shape)
        box_xy.append(K.reshape(sub_box_xy, [-1, 2]))
        box_wh.append(K.reshape(sub_box_wh, [-1, 2]))
        box_confidence.append(K.reshape(sub_box_confidence, [-1, 1]))
        box_class_probs.append(K.reshape(sub_box_class_probs, [-1, num_classes]))
    ## 将所有特征层的预测结果进行堆叠，方便进行整体处理
    box_xy          = K.concatenate(box_xy, axis = 0)
    box_wh          = K.concatenate(box_wh, axis = 0)
    box_confidence  = K.concatenate(box_confidence, axis = 0)
    box_class_probs = K.concatenate(box_class_probs, axis = 0)

    #------------------------------------------------------------------------------------------------------------#
    #   在图像传入网络预测前会进行letterbox_image给图像周围添加灰条，因此生成的box_xy, box_wh是相对于有灰条的图像的
    #   我们需要对其进行修改，去除灰条的部分。 将box_xy、和box_wh调节成y_min,y_max,xmin,xmax
    #   如果没有使用letterbox_image也需要将归一化后的box_xy, box_wh调整成相对于原图大小的
    #------------------------------------------------------------------------------------------------------------#
    ###  对图像进行调整，去除添加的灰度条 ###
    boxes       = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)
    ###将预测框的置信度和种类的置信度进行相乘###
    box_scores  = box_confidence * box_class_probs

    #-----------------------------------------------------------#
    #   判断得分是否大于score_threshold（设置好的置信度阈值）
    #-----------------------------------------------------------#
    mask             = box_scores >= confidence
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_out   = []
    scores_out  = []
    classes_out = []
    ### 对每一个种类进行非极大值抑制 ###
    for c in range(num_classes):
        #-----------------------------------------------------------#
        #   取出所有box_scores >= score_threshold的框，和成绩
        ### 即获得得分和坐标 ###
        #-----------------------------------------------------------#
        class_boxes      = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])

        #-----------------------------------------------------------#
        #   非极大抑制
        #   保留一定区域内得分最大的框
        #-----------------------------------------------------------#
        nms_index = tf.image.non_max_suppression(class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=nms_iou)

        #-----------------------------------------------------------#
        #   获取非极大抑制后的结果
        #   下列三个分别是：框的位置，得分与种类
        ###将保留后的框的：坐标，得分，种类进行返回###
        #-----------------------------------------------------------#
        class_boxes         = K.gather(class_boxes, nms_index)
        class_box_scores    = K.gather(class_box_scores, nms_index)
        classes             = K.ones_like(class_box_scores, 'int32') * c

        boxes_out.append(class_boxes)
        scores_out.append(class_box_scores)
        classes_out.append(classes)
    boxes_out      = K.concatenate(boxes_out, axis=0)
    scores_out     = K.concatenate(scores_out, axis=0)
    classes_out    = K.concatenate(classes_out, axis=0)

    return boxes_out, scores_out, classes_out


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    def sigmoid(x):
        s = 1 / (1 + np.exp(-x))
        return s
    #---------------------------------------------------#
    #   将预测值的每个特征层调成真实值
    #---------------------------------------------------#
    def get_anchors_and_decode(feats, anchors, num_classes):
        '''
                以特征层是 13*13 为例
                :param  feats:某特征层的预测结果 形状：[batch_size,13,13,3*(5+num_class)]
                                            其中：5=4+1,4：先验框调整参数 1：判断先验框内部是否包含物体
                        anchors:每一个特征层的每一个特征点的三个先验框 形状：[3,2]
                        num_classes:数据集种类个数 以coco数据集为例：80

            '''
        # feats     [batch_size, 13, 13, 3 * (5 + num_classes)]
        # anchors   [3, 2]
        # num_classes 
        # 3
        num_anchors = len(anchors) # 3
        #------------------------------------------#
        #   grid_shape指的是特征层的高和宽
        #   grid_shape [13, 13] 
        #------------------------------------------#
        grid_shape = np.shape(feats)[1:3]
        #--------------------------------------------------------------------#
        #   获得各个特征点的坐标信息。生成的shape为(13, 13, num_anchors, 2)
        #   grid_x [13, 13, 3, 1]
        #   grid_y [13, 13, 3, 1]
        #   grid   [13, 13, 3, 2]
        #--------------------------------------------------------------------#
        grid_x  = np.tile(np.reshape(np.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]), [grid_shape[0], 1, num_anchors, 1])
        grid_y  = np.tile(np.reshape(np.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[1], num_anchors, 1])
        grid    = np.concatenate([grid_x, grid_y], -1)
        #---------------------------------------------------------------#
        #   将先验框进行拓展，生成的shape为(13, 13, num_anchors, 2)
        #   [1, 1, 3, 2]
        #   [13, 13, 3, 2]
        #---------------------------------------------------------------#
        anchors_tensor = np.reshape(anchors, [1, 1, num_anchors, 2])
        anchors_tensor = np.tile(anchors_tensor, [grid_shape[0], grid_shape[1], 1, 1]) 

        #---------------------------------------------------#
        #   将预测结果调整成(batch_size,13,13,3,85)
        #   85可拆分成4 + 1 + 80
        #   4代表的是中心宽高的调整参数
        #   1代表的是框的置信度
        #   80代表的是种类的置信度
        #   [batch_size, 13, 13, 3 * (5 + num_classes)]
        #   [batch_size, 13, 13, 3, 5 + num_classes]
        #################################################################################################
        ## 对预测结果进行reshape,对原始结果的最后一维进行拆分
        ## 原始预测结果：[batch_size,13,13,3*(5+num_class)] --> reshape:[batch_size,13,13,3,5+num_class]
        ## feats:[batch_size,h,w,anchors(先验框数量)，先验框调整参数+框内部是否包含物体+物体种类]
        #################################################################################################
        #---------------------------------------------------#
        feats           = np.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])
        #------------------------------------------#
        #   对先验框进行解码，并进行归一化
        #------------------------------------------#
        ## 取出前两个维度，利用sigmoid对先验框的中心进行调整
        box_xy          = sigmoid(feats[..., :2]) + grid
        ## 对宽高进行调整  anchors_tensor：先验框的宽高
        box_wh          = np.exp(feats[..., 2:4]) * anchors_tensor
        #------------------------------------------#
        #   获得预测框的置信度
        #------------------------------------------#
        box_confidence  = sigmoid(feats[..., 4:5])
        box_class_probs = sigmoid(feats[..., 5:])
        ###   对先验框进行可视化
        box_wh = box_wh / 32
        anchors_tensor = anchors_tensor / 32
        fig = plt.figure()
        ax = fig.add_subplot(121)
        plt.ylim(-2,15)
        plt.xlim(-2,15)
        plt.scatter(grid_x,grid_y)
        plt.scatter(5,5,c='black')
        plt.gca().invert_yaxis()


        anchor_left = grid_x - anchors_tensor/2 
        anchor_top = grid_y - anchors_tensor/2 
        print(np.shape(anchors_tensor))
        print(np.shape(box_xy))
        rect1 = plt.Rectangle([anchor_left[5,5,0,0],anchor_top[5,5,0,1]],anchors_tensor[0,0,0,0],anchors_tensor[0,0,0,1],color="r",fill=False)
        rect2 = plt.Rectangle([anchor_left[5,5,1,0],anchor_top[5,5,1,1]],anchors_tensor[0,0,1,0],anchors_tensor[0,0,1,1],color="r",fill=False)
        rect3 = plt.Rectangle([anchor_left[5,5,2,0],anchor_top[5,5,2,1]],anchors_tensor[0,0,2,0],anchors_tensor[0,0,2,1],color="r",fill=False)

        ax.add_patch(rect1)
        ax.add_patch(rect2)
        ax.add_patch(rect3)

        ax = fig.add_subplot(122)
        plt.ylim(-2,15)
        plt.xlim(-2,15)
        plt.scatter(grid_x,grid_y)
        plt.scatter(5,5,c='black')
        plt.scatter(box_xy[0,5,5,:,0],box_xy[0,5,5,:,1],c='r')
        plt.gca().invert_yaxis()

        pre_left = box_xy[...,0] - box_wh[...,0]/2 
        pre_top = box_xy[...,1] - box_wh[...,1]/2 

        rect1 = plt.Rectangle([pre_left[0,5,5,0],pre_top[0,5,5,0]],box_wh[0,5,5,0,0],box_wh[0,5,5,0,1],color="r",fill=False)
        rect2 = plt.Rectangle([pre_left[0,5,5,1],pre_top[0,5,5,1]],box_wh[0,5,5,1,0],box_wh[0,5,5,1,1],color="r",fill=False)
        rect3 = plt.Rectangle([pre_left[0,5,5,2],pre_top[0,5,5,2]],box_wh[0,5,5,2,0],box_wh[0,5,5,2,1],color="r",fill=False)

        ax.add_patch(rect1)
        ax.add_patch(rect2)
        ax.add_patch(rect3)

        plt.show()
        #
    feat = np.random.normal(0,0.5,[4,13,13,75])
    anchors = [[142, 110],[192, 243],[459, 401]]
    get_anchors_and_decode(feat,anchors,20)
