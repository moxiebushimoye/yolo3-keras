# 目标识别使用（YOLO-v3）笔记

### 数据集

#### 训练集

```
整合所有已进行过人工标注的图像文件 共计3003张图像，
使用4:1的比例作为训练集和验证集分割比例用来训练模型

其他图像作为模型训练完毕识别测试（若模型精度不够则再加入新的标注图像）
```

### 训练数据集-步骤

```python
1.数据集 - VOC格式集  
	VOCdevkit 目录下 存在一下三个子文件夹
		Annotations - 存放xml格式的标签文件
		ImageSets - Main文件夹下有一下文件
			- test.txt 作为测试的文件名
			- train.txt - 所需要用来训练的图片文件索引
			- trainval.txt
             - val.txt - 验证的文件名     
		JPEGImages - 存放图片文件     
```

2.训练参数说明：

```python
# 参看train.py 文件
classes_path -- 指向数据集类别索引的txt文件
# tips: 如果是自己的数据集，可以在model_data目录下创建自己的数据集类别索引的txt文件
anchors_path -- 代表先验框对应的txt文件，一般不修改。
anchors_mask -- 用于帮助代码找到对应的先验框，一般不修改。
model_path -- 对应训练时模型所使用的预训练权重，
			一般必须要使用，如果不使用的话，训练权值会太过随机，训练结果也不会很好
input_shape  - 传入网络中的图片的输入大小，一定要是32的倍数，
			  默认 416 * 416 （ 608*608 也很常用 ）
3.训练过程（YOLO-3）
	分为两个阶段 --> 冻结阶段 -> 解冻阶段
	3.1 冻结阶段
    	-模型主干被冻结，主干特征提取网络不会发生改变
   		-由于训练参数较少，模型所占用的显存较小，仅对网络进行微调
		训练参数：
        # 默认参数：Init_Epoch = 0  Freeze_Epoch = 50
   		#默认参数下 训练 Freeze_Epoch-Init_Epoch 个世代（迭代次数） ###
    	#------------------------------------------------------------------#
    		Init_Epoch          = 0
   		 	Freeze_Epoch        = 50
    		Freeze_batch_size   = 16 ##显存较大的情况下，可以设置的更大
		    
	3.2 解冻阶段
    	- 此时模型的主干不被冻结了，特征提取网络会发生改变
   		- 训练参数较多，占用的显存较大，网络所有的参数都会发生改变  
           所以设置的batch_size 可以稍微小一点，如果电脑显存大可以设置大一些
           训练参数：
        	    训练参数 UnFreeze_Epoch      = 300：
        	    Unfreeze_batch_size = 8 
           
	Freeze_Train -- 是否进行冻结训练，如果不想开启，就设置为False,默认为True
    num_workers -- 设置是否使用多线程读取数据，1代表关闭多线程
    train_annotation_path -- 训练图片路径和标签
    val_annotation_path  -- 验证图片路径和标签
```

3.训练VOC数据集

```python
1.准备好数据集 -各目录下放入对应文件
2.打开voc_annotation.py 文件 修改相关参数
 # 如过是训练VOC数据集只需要修改	annotation_mode=2，其他参数都不需要修改
  annotation_mode - 用于指定该文件运行时计算的内容.
	0：会执行整个标签处理流程，
    	 首先会生成ImageSets里面的txt，进行训练集、验证集和测试集的划分
         根据上面的TXT在根目录下生成训练用的2007_train.txt、2007_val.txt
	1：只会生成ImageSets里面的txt，进行训练集、验证集和测试集的划分
	2：只会生成在根目录下的2007_train.txt、2007_val.txt
    # 由于voc数据集已经划分好了训练集验证集和测试集，所以这里设置为2
 3.用的时候可以删除之前生成的根目录下的文件，运行voc_annotation.py。生成相关txt文件
 4.生成完毕，打开train.py文件直接运行，不需要修改参数（本代码默认训练voc数据集）
    
```

4.训练自己的数据集

```python
1.准备好数据集
	数据集包括：Annotations - 存放的xml标签文件
    		  JPEGImages - 存放图片文件
2.打开voc_annotation.py 文件 修改相关参数
#如果是第一次训练自己的数据集，只需要修改 classes_path 即可
########################################################
# 其他参数注解
trainval_percent  用于指定(训练集+验证集)与测试集的比例，默认情况下 (训练集+验证集):测试集 = 9:1
train_percent用于指定(训练集+验证集)中训练集与验证集的比例，默认情况下 训练集:验证集 = 9:1   # trainval_percent和train_percent 仅在annotation_mode为0和1的时候有效   
VOCdevkit_path - 指向VOC数据集所在的文件夹，默认指向根目录下的VOC数据集
########################################################
3.运行voc_annotation.py。生成相关txt文件   
4.打开 train.py文件，修改classes_path为自己数据集model_data下的txt文件的路径
5.运行train.py文件

```

5.利用自己训练好的模型进行预测

```python
1.参数修改（class YOLO（）中）
# yolo.py文件
"第一训练，一定要修改其中的model_path和classes_path"
model_path -- 对应模型权值文件的路径
classes_path -- 对应model_data下的txt文件（和训练时的一样）
2.运行predict.py 
  输入图片路径，查看识别效果
#################################################
# 其他参数说明
	anchors_path - 代表先验框对应的txt文件，一般不修改。
	anchors_mask - 用于帮助代码找到对应的先验框，一般不修改。
    input_shape - 预测图片的大小，默认 416 * 416，必须是32的倍数
    confidence - 置信度阈值，只有大于confidence的先验框才会被保留下来
    nms_iou - 非极大抑制所用到的nms_iou大小，值越小代表非极大值抑制越严格
    max_boxes - 最大目标数量，默认100
    letterbox_image - True/False 用于判断是否需要使用不失真的resize
#################################################
```

6.利用自己训练好的模型进行mAP评估

```python
1.评估前要按照predict的步骤修改对应的model_path classes_path
2.完成后进入 get_map.py文件，修改参数
  # 第一次进行评估，只需要修改classes_Path，其他参数不需要修改
3.运行get_map.py即可
#################################################
# 其他参数说明
map_mode - 指定该文件运行时计算的内容
	0:完成整个map计算流程：①获得测试集的预测结果 ②获得测试集对应的真实框 ③根据①②计算map
    1： 获得测试集的预测结果
    2:  获得测试集对应的真实框
    3:  计算map
    4:  利用COCO工具箱计算当前数据集的0.5~0.95map的评估结果。
        # 需要安装pycocotools才行
classes_path
MINOVERLAP - 指定想要获得的mAP的iou的值
map_vis - 用于指定是否开启VOC_map计算的可视化 True/False
VOCdevkit_path -指定voc所在文件夹
map_out_path - 结果输出文件夹
#################################################
```

7.yolov3网络在训练前是如何进行预处理的

```python
# path:utils--dataloader.py
重点关注： __getitem__ 函数

```

8.损失计算过程

```python
# path:nets--yolo_training.py -- yolo_loss
1.将预测结果和真实结果进行划分 y_true:真实结果，yolo_output:预测结果
2.获得输入图片的大小（input_shape）：默认 416*416
3.获得网格的shape (实际就是： 13*13 26*26 52*52)
4.取出batch_size的大小
5.对每一个特征层进行循环，循环内部对每一个特征层进行单独处理：
  5.1 取出 y_true中特征层是否包含特征点的mask
  5.2 获取每个特征点对应的种类  
  5.3 利用yolo网络的预测结果，对先验框进行调整，获得预测框的 w h x y
    	#   grid        (13,13,1,2) 网格坐标
        #   raw_pred    (m,13,13,3,85) 尚未处理的预测结果
        #   pred_xy     (m,13,13,3,2) 解码后的中心坐标
        #   pred_wh     (m,13,13,3,2) 解码后的宽高坐标
   5.4 获得先验框调整后的预测结果（x,y,w,h）后，将预测结果进行堆叠（K.concatenate）
   5.5 完成堆叠后，调用loop_body（）
		1.将预测框和真实框取出，计算他们之间的IOU
		2.判断每个特征点与真实框的最大重合程度（最大iou）
         3.判断预测框和真实框的最大iou小于ignore_thresh，则认为该预测框没有与之对应的真实框
           [目的：忽略预测结果与真实框非常对应特征点，因为这些框已经比较准了，不适合当作负样本]
  6. 循环结束，对上面的循环结果进行堆叠（stack()）
  7. 将真实框进行编码，用真实框的结果-grid(网格点的坐标)，从而获得网络在第0个序号和第1序号应该有的预测结果
  8.获得网络宽高应该有的调整的结果
  9.根据获取到的真实框内部是否包含物体，对真实框编码后的结果进行选取，计算
    [比较小真实框的权值调大一点，比较大真实框的权值调小一点]
  10.利用交叉熵（binary_crossentropy） 计算x，y轴的偏移情况（xy_loss）
  11.计算有目标的特征点里 w,h 的损失 
  12.利用交叉熵计算置信度的损失
  13.利用交叉熵计算种类的损失
  14.将计算的所有损失（xy,wh,confidence，class）进行求和,得到loss
  15.计算正样本的数量
  16.loss/正样本数量 -->获得最终的损失
```

