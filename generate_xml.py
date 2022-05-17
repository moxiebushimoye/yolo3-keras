import os
from PIL import Image
from yolo import YOLO


def xml_main(target_dir,xml_path):
    if not os.path.exists(xml_path):
        os.makedirs(xml_path)
    # img_path = "appleimg/apple1.jpg"
    imgpath_list=os.listdir(target_dir)
    yolo = YOLO()
    for img_name in imgpath_list:
            image_path = os.path.join(target_dir, img_name)
            image = Image.open(image_path)
            # crop = False
            # count = False
            # 调用模型预测
            yolo.get_xml(image_path, image, xml_path)


if __name__ == '__main__':
    # 存储待标注图像的文件夹路径
    target_dir = "appleimg/"
    # 生成标志文件的存储路径
    xml_path = "xml_anno/"
    xml_main(target_dir,xml_path)