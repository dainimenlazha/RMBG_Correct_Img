# 项目背景

需要一个自动化处理图片拼接组合的脚本，第一步就需要分离出主体，并转成透视正确的图片。
找了很多开源实现，效果都不是很理想，要么是透视结果错误，要么是寻点结果报错，所以自己写一个。

# 运行说明

需要下载RMBG到本地运行，暂时没有CUDA，运行速度较慢，使用CPU计算。
[RMBG传送门: https://huggingface.co/briaai/RMBG-2.0/tree/main](https://huggingface.co/briaai/RMBG-2.0/tree/main)


# 实现原理

运用rmbg和opencv实现对复杂背景中主体画面的透视矫正。
整体思路是：
运用RMBG检测出主体--》运用opencv的边缘检测--》通过最大面积找点--》通过K Means算法划分成四个部分（左上，右上，左下，右下）--》获取每个区域内距离中心最远的点--》运用opencv的透视变换

# 实际效果

![image](https://github.com/user-attachments/assets/163c34c8-bc82-4072-b58b-9041516c35b8)

![image](https://github.com/user-attachments/assets/8342f81e-1a8e-43bf-b8fe-e2d3afae7c6b)

![image](https://github.com/user-attachments/assets/bf577444-e89f-4846-ab9e-a3554dc7d344)

![image](https://github.com/user-attachments/assets/ca87f856-a075-47ef-85c9-6b8967e9f8cb)

# TODO

1、目前有个问题，只能透视修正主体为矩形的图片，但已经满足我的需求了；目前的瓶颈在边缘检测上，一直不满意，空了再研究吧。

2、想实现一个可以自动美化图片的算法，主要方向是提升亮度，饱和度，对比度等
