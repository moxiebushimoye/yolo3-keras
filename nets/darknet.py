from functools import wraps

from keras.initializers import random_normal
from keras.layers import (Add, BatchNormalization, Conv2D, LeakyReLU,
                          ZeroPadding2D)
from keras.regularizers import l2
from utils.utils import compose


#---------------------------------------------------#
#   单次卷积
#   DarknetConv2D - 参照kerasP85页代码内容
    ## Conv2D
    # 第一个参数是过滤器的数量，元祖是每个过滤器的大小
#---------------------------------------------------#
@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    darknet_conv_kwargs = {'kernel_initializer' : random_normal(stddev=0.02)}
    # padding:有两个选项
        # valid：仅在输入和过滤器完全重叠，输出小于输入的情况的下计算卷积
        # same： 输入与输出大小相同，输入以外的区域全部用0填充
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)

#---------------------------------------------------#
#   卷积块 -> 卷积 + 标准化 + 激活函数
#   DarknetConv2D + BatchNormalization + LeakyReLU
#---------------------------------------------------#
def DarknetConv2D_BN_Leaky(*args, **kwargs):
    '''
    调动本函数，完成一个卷积 + 标准化 + 激活函数 的过程
    '''
    no_bias_kwargs = {'use_bias': False} #？
    no_bias_kwargs.update(kwargs)
    return compose( 
        DarknetConv2D(*args, **no_bias_kwargs), # 卷积
        BatchNormalization(), # 标准化
        LeakyReLU(alpha=0.1)) # 激活函数

#---------------------------------------------------------------------#
#   残差结构
#   首先利用ZeroPadding2D和一个步长为2x2的卷积块进行高和宽的压缩
#   然后对num_blocks进行循环，循环内部是残差结构。
    ## keras.layers.convolutional.ZeroPadding2D(padding=((1,0),(1,0)))
    ## 在第一行前面加一行零,第一列前面加一列零。行数增1,列数增1
    #  在每一次完成resblock_body后,特征层的宽和高会变为原来的1/2
#---------------------------------------------------------------------#
def resblock_body(x, num_filters, num_blocks):
    '''
        构建 resblock_body
        :param  x: 输入
                num_filters:过滤器数量
                num_blocks:残差结构的堆叠次数
        :return x:输出
    '''
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3,3), strides=(2,2))(x)
    # 进行残差结构的堆叠，由num_blocks的数量决定
    for i in range(num_blocks):
        y = DarknetConv2D_BN_Leaky(num_filters//2, (1,1))(x) # 利用一个1*1的卷积进行通道数的下降
        y = DarknetConv2D_BN_Leaky(num_filters, (3,3))(y)   # 利用3*3的卷积进行特征提取
        x = Add()([x,y]) # 将特征结果和残差边进行相加
    return x

#---------------------------------------------------#
#   darknet53 的主体部分
#   输入为一张416x416x3的图片
#   输出为三个有效特征层
#---------------------------------------------------#
def darknet_body(x):
    '''
        :param x: 输入的图片，大小为 416*416*3
            1.通过DarknetConv2D_BN_Leaky完成一次卷积+标准化+激活函数，获取到shape为416*416*32的特征层
            2.调用resblock_body，构建resblock
        :return:feat1：
                feat2：
                feat3：
    '''
    # 416,416,3 -> 416,416,32
    x = DarknetConv2D_BN_Leaky(32, (3,3))(x)
    # 416,416,32 -> 208,208,64
    x = resblock_body(x, 64, 1)
    # 208,208,64 -> 104,104,128
    x = resblock_body(x, 128, 2)
    # 104,104,128 -> 52,52,256
    x = resblock_body(x, 256, 8)
    feat1 = x
    # 52,52,256 -> 26,26,512
    x = resblock_body(x, 512, 8)
    feat2 = x
    # 26,26,512 -> 13,13,1024
    x = resblock_body(x, 1024, 4)
    feat3 = x
    return feat1, feat2, feat3

