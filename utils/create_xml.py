from xml.dom import minidom
import xml.etree.ElementTree as ET
# from yolo import YOLO
# import time
#
# import cv2
# import numpy as np
# from PIL import Image

class VOC_Sample_Generator:

    def __init__(self):
        self.root_node = ET.Element('annotation')
        # self.filename_node = ET.SubElement(self.root_node, "filename")
        # self.size_node=ET.SubElement(self.root_node,"size")
        # self.object_node = ET.SubElement(self.root_node,"object")
        # self.filename_node.text=filename


    def add_filename(self,filename):
        '''
        创建filename部分元素内容
        '''
        self.filename_node = ET.SubElement(self.root_node, "filename")
        self.filename_node.text = filename

    def add_size(self, width, height, depth):
        '''
        创建size部分元素内容
        '''
        self.size_node = ET.SubElement(self.root_node, "size")
        width_node=ET.SubElement(self.size_node,"width")
        width_node.text=str(width)
        height_node = ET.SubElement(self.size_node, "height")
        height_node.text=str(height)
        depth_node = ET.SubElement(self.size_node, "depth")
        depth_node.text=str(depth)

    def __add_class(self, class_name):
        '''
        创建name部分元素内容
        '''
        name_node=ET.SubElement(self.object_node,"name")
        name_node.text=class_name


    def __add_bndbox(self,xmin, ymin, xmax, ymax):
        '''
        创建bndbox部分元素内容
        '''
        xmin_node = ET.SubElement(self.bndbox_node,"xmin")
        xmin_node.text=str(xmin)
        ymin_node = ET.SubElement(self.bndbox_node,"ymin")
        ymin_node.text=str(ymin)
        xmax_node = ET.SubElement(self.bndbox_node,"xmax")
        xmax_node.text=str(xmax)
        ymax_node = ET.SubElement(self.bndbox_node,"ymax")
        ymax_node.text=str(ymax)



    def build_object(self,class_name,xmin, ymin, xmax, ymax):
        # 创建object结点内部元素-name,bndbox,pose
        self.object_node=ET.SubElement(self.root_node,"object")
        self.bndbox_node=ET.SubElement(self.object_node,"bndbox")
        self.__add_class(class_name)
        pose_node=ET.SubElement(self.object_node,"pose")
        pose_node.text="Unspecified"
        truncated=ET.SubElement(self.object_node,"truncated")
        truncated.text="0"
        self.__add_bndbox(xmin, ymin, xmax, ymax)



    def build(self, path):
        '''
        创建xml文件
        '''
        xml_str=ET.tostring(self.root_node,'utf-8')
        reparse=minidom.parseString(xml_str)
        newstr=reparse.toprettyxml(indent='\t')
        with open(path,"w",encoding="utf-8") as f:
            f.write(newstr)
        f.close()




if __name__ == '__main__':
    img_name="test.jpg"
    voc = VOC_Sample_Generator()
    voc.add_filename(img_name)
    voc.add_size(550, 550, 3)
    for i in range(5):
        voc.build_object("appl",i,6,7,8)
    voc.build("test.xml")


