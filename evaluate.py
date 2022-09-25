import numpy as np
import argparse
import json
from PIL import Image
from os.path import join
import os
#设标签宽W，长H
def fast_hist(a, b, n):
    # a是转化成一维数组的标签，形状(H×W,)，即每个像素的GT label
    # b是转化成一维数组的标签，形状(H×W,)；pred label,n是类别数目，实数（在这里为19）
    '''
	核心代码
	'''
    k = (a >= 0) & (a < n)
    # k是一个一维bool数组，形状(H×W,)；去掉了a中数值（类别）无效的标签
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)
    # a[k]是将a中有效的值提出来，而b[k]是在和a对应的k矩阵上也提出值，即和a的位置是一一对应的
    # n*a+b是利用类别映射成了index,例如a=0&b=0时应该在第0位置上，a=1&b=0应该在n位置上
    # np.bincount计算了每个index出现的次数，并按照index大小排列成n*n矩阵，这个函数其实就是拿GT和pred组成了混淆矩阵
 
 
def per_class_iu(hist):
    # 分别为每个类别（在这里是19类）计算IoU，hist即n*n混淆矩阵
    '''
	核心代码
	'''
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    # IoU的计算方法，TP(对角线之和)/(TP+FN(每一行之和)+TP+FP(每一列之和)-多算了一遍的TP)，返回值形状(n,)，每个值为每个类的IoU
 
 
def label_mapping(input, mapping):
    # 主要是因为CityScapes标签里面原类别太多，这样做把其他类别转换成算法需要的类别（共19类）和背景（标注为255）
    # mapping是34*2的矩阵，34是原标签类别
    output = np.copy(input)
    for ind in range(len(mapping)):
        output[input == mapping[ind][0]] = mapping[ind][1]
        # 进行类别映射，mapping[ind][0]为原类别的数值，[ind][1]为0~18的标签数值
        # input == mapping[ind][0]是一个形状和input一样的bool数组，
        # 即input中和原类别相等位置上的值，用mapping中0~18的值替换掉了
    return np.array(output, dtype=np.int64)
'''
compute_mIoU函数是以CityScapes图像分割验证集为例来计算mIoU值的
由于作者个人贡献的原因，本函数除了最主要的计算mIoU的代码之外，还完成了一些其他操作，
比如进行数据读取，因为原文是做图像分割迁移方面的工作，因此还进行了标签映射的相关工作，在这里笔者都进行注释。
大家在使用的时候，可以忽略原作者的数据读取过程，只需要注意计算mIoU的时候每张图片分割结果与标签要配对。
主要留意mIoU指标的计算核心代码即可。
'''
def compute_mIoU(gt_dir, pred_dir):#计算mIoU的函数
    """
    Compute IoU given the predicted colorized images and 
    """
    num_classes = 19 #读取类别数目，这里是19类，详见博客中附加的info.json文件
    print('Num classes', num_classes)#打印一下类别数目
    name_classes = np.array([
      "road",
      "sidewalk",
      "building",
      "wall",
      "fence",
      "pole",
      "light",
      "sign",
      "vegetation",
      "terrain",
      "sky",
      "person",
      "rider",
      "car",
      "truck",
      "bus",
      "train",
      "motocycle",
      "bicycle"], dtype=np.str)#读取类别名称，详见博客中附加的info.json文件
    hist = np.zeros((num_classes, num_classes))#hist初始化为全零，在这里的hist的形状是[19, 19]
 
    # image_path_list = join(devkit_dir, 'val.txt')#在这里打开记录验证集图片名称的txt
    # label_path_list = join(devkit_dir, 'label.txt')#在这里打开记录验证集标签名称的txt
    # gt_imgs = open(label_path_list, 'r').read().splitlines()# 获得GT图片名称列表
    img_names = sorted(os.listdir(gt_dir))
    gt_imgs = [join(gt_dir, x) for x in img_names]# 获得GT图片路径列表，方便直接读取

    # pred_imgs = open(image_path_list, 'r').read().splitlines()#获得验证集图像分割结果名称列表
    pred_imgs = [join(pred_dir, x) for x in img_names]#获得验证集图像分割结果路径列表，方便直接读取
 
    for ind in range(len(gt_imgs)):#读取每一个（图片-标签）对
        pred = np.array(Image.open(pred_imgs[ind]))#读取一张图像分割结果，转化成numpy数组
        label = np.array(Image.open(gt_imgs[ind]))#读取一张GT对应的标签，转化成numpy数组
        # label = label_mapping(label, mapping)#进行标签映射（因为没有用到全部类别，因此舍弃某些类别），可忽略
        if len(label.flatten()) != len(pred.flatten()):#如果图像分割结果与GT标签的大小不一样，这张图片就不计算
            print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()), len(pred.flatten()), gt_imgs[ind], pred_imgs[ind]))
            continue
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)#对一张图片计算19×19的hist矩阵，并累加
        if ind > 0 and ind % 10 == 0:#每计算10张就输出一下目前已计算的图片中所有类别平均的mIoU值
            print('{:d} / {:d}: {:0.2f}'.format(ind, len(gt_imgs), 100*np.mean(per_class_iu(hist))))
    
    mIoUs = per_class_iu(hist)#计算所有验证集图片的逐类别IoU值(每个类别的IoU)
    for ind_class in range(num_classes):#逐类别输出一下mIoU值
        print('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))#在所有验证集图像上求所有类别平均的mIoU值，计算时忽略NaN值
    return mIoUs
 
 
def main(args):
   compute_mIoU(args.gt_dir, args.pred_dir)#执行计算mIoU的函数
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('gt_dir', type=str, help='directory which stores CityScapes val gt images')#设置gt_dir参数，存放验证集分割标签的文件夹
    parser.add_argument('pred_dir', type=str, help='directory which stores CityScapes val pred images')#设置pred_dir参数，存放验证集分割结果的文件夹
    args = parser.parse_args()
    main(args)#执行主函数