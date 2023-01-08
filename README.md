# queueOYL_value(编辑中)

在这里我们将用tensorflow调用keras来完成模型的搭建，用mnist模型训练并且验证

1. mnist数据的导入与处理，不过多阐述
      (minst：包含6w张数字图片，图片规格为28*28，映射到个位数上)
      
2. 开始构建神经网络的框架：
      a.  因为数据集为图形，我们可以理解为矩阵(28*28)
          
      b.  在选择网络层api时，需要注意输入层的数据格式，
          我们这里选择 全连接层(Dense), 但Dense接受数据需要为一维‘向量’,
          所以我们需要使用 Flatten 对数据进行降维
          
      c.  Flatten: 用input_shape(28,28)传入矩阵后，将矩阵降维为‘向量’，
          这样Dense的数据形式就
          


