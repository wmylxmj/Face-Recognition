# Face-Recognition
基于Siamese网络的人脸识别 2019-3-9
### 数据集的准备
- 将CAS-PEAL-R1数据集放入datasets文件夹
- 建立FaceRecognition的类实例
- 调用prepare方法进行数据处理
### CAS-PEAL-R1 数据集目录
    +-<CAS-PEAL-R1>			... The root directory of this CDROM.
        |
        |
        +-<Documents>		... The documentation directory. In this directory the pdf version of
        |				    the technical report of CAS-PEAL database and the CAS-PEAL
        |			 	    Database Release Agreement can be found.
        |
        +-<Evaluation Prototype>	... This directory contains the evaluation prototype of the 
        |   |                           CAS-PEAL-R1 database.
        |	|
        |	+-<Gallery>		... This directory contains one text file which records the image names  
        |   |                           in the Galley.
        |	|				    
        |	|
        |	+-<Probe Sets>		... This directory contains seven text files. Each text file records 
        |   |                           the image names of a specific probe set.
        |	|			   
        |	|
        |	+-<Training Set>	... This directory contains one text file which records the image names 
        |                               in the Training set.
        |			   	   
        |	
        +-<FRONTAL>			... Frontal Subset of CAS-PEAL-R1
        |	|
        |	+-<Accessory>		... images with accessory variations
        |	|
        |	+-<Background>		... images with background variations
        |	|
        |	+-<Distance>		... images with distance variations
        |	|
        |	+-<Expression>		... images with expression variations
        |	|
        |	+-<Lighting>		... images with lighting variations
        |	|
        |	+-<Aging>		... images with aging variations
        |	|
        |	+-<Normal>		... images under normal conditions 
        |
        +-<POSE>			... Pose Subset of CAS-PEAL-R1.
        	|
        	+-<000001>		... All the pose images of one person are stored in one directory
        	|
        	+-<000002>		... Each directory's name is one person's ID.
        	|
        	...
        	|
            +-<001042>
### 训练模型
- 建立FaceRecognition的类实例
- 调用train方法进行训练
- 若载入权重，将load_pretrained置为True
### 预测模型
- 建立FaceRecognition的类实例
- 调用predict方法进行预测，输入为图片文件路径，输出1为同一人，输出0为不同人
- 训练好的权重下载地址
- 链接：https://pan.baidu.com/s/1YinGcQ8_4FKRKrIo0fd7Xw 提取码：7iog 
### 关于模型
- 借鉴ResNet50
- 采用多个小卷积核级联的形式提高感受野
- 使用inception block形式将不同感受野结合
- 采用leaky relu加快训练
- 使用L1距离加上logistic回归单元进行预测
### 模型图片
![](images/SiameseNetwork.png)
