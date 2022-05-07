from keras.layers import Concatenate, Input, Lambda, UpSampling2D
from keras.models import Model
from utils.utils import compose

from nets.darknet import DarknetConv2D, DarknetConv2D_BN_Leaky, darknet_body
from nets.yolo_training import yolo_loss

#---------------------------------------------------#
#   特征层->最后的输出
#---------------------------------------------------#
def make_five_conv(x, num_filters):
    '''
    对特征进行5次卷积 - 实际上是一个不断下降通道数，然后进行特征提取的过程
    利用 1*1的卷积核进行通道下降；利用3*3的卷积核进行特征提取
    '''
    x = DarknetConv2D_BN_Leaky(num_filters, (1,1))(x)
    x = DarknetConv2D_BN_Leaky(num_filters*2, (3,3))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (1,1))(x)
    x = DarknetConv2D_BN_Leaky(num_filters*2, (3,3))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (1,1))(x)
    return x

def make_yolo_head(x, num_filters, out_filters):
    '''
    构建yolo_head:
        将获取到的有效特征层转换为这个特征层所对应的预测结果
        Yolo Head本质上是一次3x3卷积加上一次1x1卷积，
        先用3x3卷积核进行特征融合，再用1x1卷积核调整通道数
    '''
    y = DarknetConv2D_BN_Leaky(num_filters*2, (3,3))(x)
    ## 如果是coco数据集，这个通道数是255 --> 可拆分为：3*85。3代表每一个特征层对应的特征点所具有的三个先验框
    ## 85可以拆分为：4+1+80 4：先验框调整参数 1：判断先验框内部是否包含物体 80：判断先验框内物体所对应的物体种类
    y = DarknetConv2D(out_filters, (1,1))(y)
    return y

#---------------------------------------------------#
#   FPN网络的构建，并且获得预测结果
#---------------------------------------------------#
def yolo_body(input_shape, anchors_mask, num_classes):
    '''
    构建FPN网络，对C3,C4,C5进行卷积和上采样
    将5次卷积完成后的C3,C4,C5传入到yolohead获得预测结果
    '''
    inputs      = Input(input_shape)
    #---------------------------------------------------#   
    #   生成darknet53的主干模型
    #   获得三个有效特征层，他们的shape分别是：
    #   C3 为 52,52,256
    #   C4 为 26,26,512
    #   C5 为 13,13,1024
    #---------------------------------------------------#
    C3, C4, C5  = darknet_body(inputs)

    #---------------------------------------------------#
    #   第一个特征层
    #   y1=(batch_size,13,13,3,85)
    #---------------------------------------------------#
    # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512
    ### 对C5进行5次卷积
    x   = make_five_conv(C5, 512)
    ### 将C5的5次卷积结果，传入到yolo_head（）中，获得该尺度的预测结果
    P5  = make_yolo_head(x, 512, len(anchors_mask[0]) * (num_classes+5))
    ### 对13*13的特征进行一次卷积和上采样，得到26*26的特征
    # 13,13,512 -> 13,13,256 -> 26,26,256
    x   = compose(DarknetConv2D_BN_Leaky(256, (1,1)), UpSampling2D(2))(x)
    ### 讲获取到的上采样结果（26*26*512特征）和C4（26*26*256特征）进行堆叠（拼接）
    # 26,26,256 + 26,26,512 -> 26,26,768
    x   = Concatenate()([x, C4])
    #---------------------------------------------------#
    #   第二个特征层
    #   y2=(batch_size,26,26,3,85)
    #---------------------------------------------------#
    # 26,26,768 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
    ### 进行五次卷积；然后进行一次卷积核上采样；将上采样的结果和C3进行堆叠
    x   = make_five_conv(x, 256)
    ### 将拼接后进行5次卷积的结果，传入到yolo_head（）中，获得该尺度的预测结果
    P4  = make_yolo_head(x, 256, len(anchors_mask[1]) * (num_classes+5))

    # 26,26,256 -> 26,26,128 -> 52,52,128
    x   = compose(DarknetConv2D_BN_Leaky(128, (1,1)), UpSampling2D(2))(x)
    # 52,52,128 + 52,52,256 -> 52,52,384
    x   = Concatenate()([x, C3])
    #---------------------------------------------------#
    #   第三个特征层
    #   y3=(batch_size,52,52,3,85)
    #---------------------------------------------------#
    # 52,52,384 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128
    ### 进行5次卷积，完成特征金子塔的构建
    x   = make_five_conv(x, 128)
    ### 将拼接后进行5次卷积的结果，传入到yolo_head（）中，获得该尺度的预测结果
    P3  = make_yolo_head(x, 128, len(anchors_mask[2]) * (num_classes+5))
    ### 解码过程：将预测结果转换为预测框显示在图片上
    return Model(inputs, [P5, P4, P3])


def get_train_model(model_body, input_shape, num_classes, anchors, anchors_mask):
    y_true = [Input(shape = (input_shape[0] // {0:32, 1:16, 2:8}[l], input_shape[1] // {0:32, 1:16, 2:8}[l], \
                                len(anchors_mask[l]), num_classes + 5)) for l in range(len(anchors_mask))]
    model_loss  = Lambda(
        yolo_loss, 
        output_shape    = (1, ), 
        name            = 'yolo_loss', 
        arguments       = {
            'input_shape'       : input_shape, 
            'anchors'           : anchors, 
            'anchors_mask'      : anchors_mask, 
            'num_classes'       : num_classes, 
            'balance'           : [0.4, 1.0, 4],
            'box_ratio'         : 0.05,
            'obj_ratio'         : 5 * (input_shape[0] * input_shape[1]) / (416 ** 2), 
            'cls_ratio'         : 1 * (num_classes / 80)
        }
    )([*model_body.output, *y_true])
    model       = Model([model_body.input, *y_true], model_loss)
    return model
