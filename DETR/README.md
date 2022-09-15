<!--
 * @Author: kavinbj
 * @Date: 2022-09-13 17:50:02
 * @LastEditTime: 2022-09-14 21:46:34
 * @FilePath: README.md
 * @Description: 
 * 
 * Copyright (c) 2022 by kavinbj, All Rights Reserved. 
-->

# DETR 学习笔记


### 《End-to-End Object Detection with Transformers》
论文地址：https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460205.pdf

pytorch版本代码：https://github.com/facebookresearch/detr


1. 将目标检测看作预测框和类别的集合问题
2. DETR直接预测的方式去解决目标检测的问题，绕过了所有人工设计的部分，比如NMS和anchor。

- End-to-End 检测
- Set Prediction 思想
- Anchor-Free
- Transformer 学习图像块之间的Interaction
- parallel decode 
- 

### DETR核心要点
- DETR不使用proposal和anchor，直接使用transformer全局建模的能力，把目标检测作为集合预测的问题，因为拥有了全局建模的能力，不会输出冗余的框，直接输出结果
- 提出了一个新的匹配函数，通过二分图匹配的方式，强制一个物体输出一个框
- 使用了Transformer encoder decoder的架构，transformer解码器有另外一个输入 learned object queries，有点类似与anchor，DETR可以将learned object queries和全局的图像信息结合到一起，通过不断自注意力操作，从而让模型直接并行输出一组预测框。
  
# DETR structure
![DETR structure](https://pic3.zhimg.com/v2-b37be5d54810e8ead4364b13da53f440_1440w.jpg?source=172ae18b)


### 存在的问题
- 小物体检测性能不好
- 训练时间长



### 基于集合的目标函数：N个输出与Ground truth匹配（二分图匹配）
- 简单可以使用遍历，所有的排列组合跑一边，但是这样计算复杂度比较；
- 匈牙利算法是一个高效的算法，scipy包里面有一个linear-sum-assignment，输入cost matrix（损失矩阵），得到最优排列；
- 论文中使用上述函数，损失矩阵中放的就是loss，即分类损失和框的回归损失。


Set-based loss vs proposals vs anchors

object query

object detection set prediction loss

最优二分图匹配 cost matrix 匈牙利算法
寻找最优匹配，跟原来目标检测方法利用人的先验知识，去把预测值和proposal或者anchor做匹配的方式，意思差不多。只是约束更强，一定要得到一对一的对应关系，而不是一对多。


Deformable DETR 解决了DETR的缺陷
