# Face-Recognition
基于Siamese网络的人脸识别 2019-3-9
### 数据集的准备
- 将CAS-PEAL-R1数据集放入datasets文件夹
- 建立FaceRecognition的类实例
- 调用prepare方法进行数据处理
### 训练模型
- 建立FaceRecognition的类实例
- 调用train方法进行训练
- 若载入权重，将load_pretrained置为True
### 预测模型
- 建立FaceRecognition的类实例
- 调用predict方法进行预测，输入为图片文件路径，输出1为同一人，输出0为不同人
### 关于模型
- 借鉴ResNet50
- 采用多=小卷积核级联的形式提高感受野
- 使用inception block形式将不同感受野结合
- 采用leaky relu加快训练
- 使用L1距离加上logistic回归单元进行预测
### 模型图片
![](images/SiameseNetwork.png)
