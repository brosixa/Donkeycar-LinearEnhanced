# Donkeycar项目扩展之KerasLinear模型优化

Donkeycar-LinearEnhanced源自大二时参加的个性化实验项目，对Donkeycar原本KerasLinear模型的改进



## 一、现有问题

根据官方文档描述，KerasLinear模型的主要缺点是"有时可能无法很好地学习油门控制"。

原文地址：[donkeydocs/docs/parts/keras.md at master · autorope/donkeydocs](https://github.com/autorope/donkeydocs/blob/master/docs/parts/keras.md)

模型源代码：[donkeycar/donkeycar/parts/keras.py at main · autorope/donkeycar](https://github.com/autorope/donkeycar/blob/main/donkeycar/parts/keras.py)

通过分析代码，原本KerasLinear模型的结构如下：

- **输入**：图像，形状为 `(120, 160, 3)`
- **特征提取**：5 个卷积层（具体实现为 `core_cnn_layers`）
- **全连接层**：
  - Dense(100, relu)
  - Dense(50, relu)
- **输出**：2 个线性输出（steering 和 throttle）
- **激活函数**：ReLU + Linear
- **损失函数**：MSE（均方误差）
- **优化器**：默认 Adam，可选 RMSProp、SGD



### 我发现以下可以改进的点：

1. 模型结构较为简单。只有两个全连接层(100和50个神经元)，神经元数量太少，容易欠拟合。

2. 没有充分利用现代CNN架构的优势。模型中`core_cnn_layers` 是 5 层简单堆叠的卷积 + Dropout，学习能力差。

3. 激活函数ReLU存在容易导致“神经元死亡”的问题。如果一个神经元的输出长期落在x≤0这个区间，它的权重将 无法更新，变成“死神经元”。

4. 对于连续值输出的损失函数仅使用MSE。MSE对异常值敏感，误差大时平方会放大，如果数据不平衡，预测偏离严重的情况惩罚过重。模型需要通过训练学习决策两个值：油门（Throttle）和转向（Steering），其中油门本身属于震荡幅度较大的值，而转向却需要精准控制，容忍小误差但惩罚大误差。

5. 可进行感兴趣区域（Region Of Interest, ROI）处理。我在另一做扩展内容用OpenCV结合PID算法实现纯视觉非神经网络实现小车在模拟环境中自动驾驶时发现，与现实生活中类似，决策汽车油门和转向时，主要关注道路的曲直程度。

   如图所示：

   ![test](.\image\road.jpg)

   ![lined_img](.\image\lined_img.jpg)

   而画面的天空、周围的土地和远处的马路，对当下转向和油门决策影响较小。



## 二、优化方案

#### 1.  卷积层（CNN）升级

- 更现代的架构：5 层 CNN，分别为 24→36→48→64→64 filters，提高学习能力，防止欠拟合

- 使用 `ELU` 激活函数 替代 ReLU，有助于减少梯度消失

  ![elu_vs_relu](.\image\elu_vs_relu.png)

  上图展示了 ReLU 和 ELU 激活函数的对比：

  - **蓝线（ReLU）**：在x≤0 时输出恒为 0，导致对应神经元梯度为 0（可能“死亡”）。
  - **橙线（ELU）**：在 x≤0 时逐渐趋近于 −α（此处假设α=1），输出与梯度都为非零值，避免梯度消失。

- 加入 `BatchNormalization`：提高训练稳定性和收敛速度

- 引入 `Dropout (0.3)`：随机临时丢弃部分神经元，防止过拟合

#### 2.  全连接层增强

- 增加深度：Dense(256 → 128 → 64)，激活函数为 `ELU`
- 每层均使用 `Dropout (0.3)`

#### 3.  输出层设计

- 输出仍为两个标量（steering + throttle）
- 每个输出通过 `Dense(1, linear)` 独立回归

#### 4.  损失函数增强

- steering 输出：使用 MSE，平滑控制角度
- throttle 输出：使用 Huber loss，鲁棒控制油门
- 损失权重：steering（0.7），throttle（0.3）

#### 5. 训练优化器

- 明确指定使用 `Adam` 优化器，学习率自适应ROI

#### 6. 增加空间注意力模块和图像ROI预处理

- 增强模型关注车道关键区域
- 训练前只输入图像的中下部分，重构run()函数增添ROI操作



## 三、优化后的模型代码 KerasLinearEnhanced.py

```python
class KerasLinearEnhanced(KerasPilot):
    """
    增强版的KerasLinear模型，改进油门控制能力
    """
    def __init__(self,
                 interpreter: Interpreter = KerasInterpreter(),
                 input_shape: Tuple[int, ...] = (120, 160, 3),
                 num_outputs: int = 2):
        super().__init__(interpreter, input_shape)
        self.num_outputs = num_outputs
        self.optimizer = "adam"  # 使用Adam优化器

    def create_model(self):
        return self.enhanced_linear_model(self.num_outputs, self.input_shape)

    def compile(self):
        # 使用混合损失函数 - MSE用于角度，Huber用于油门(对异常值更鲁棒)
        losses = {
            'n_outputs0': 'mse',  # 转向角度
            'n_outputs1': 'huber'  # 油门
        }
        loss_weights = {'n_outputs0': 0.7, 'n_outputs1': 0.3}  # 更重视转向
        self.interpreter.compile(
            optimizer=self.optimizer,
            loss=losses,
            loss_weights=loss_weights
        )

    def enhanced_linear_model(self, num_outputs, input_shape):
        """增强的线性模型架构"""
        drop = 0.3
        img_in = Input(shape=input_shape, name='img_in')
        
        # 使用更现代的CNN架构
        x = Convolution2D(24, (5,5), strides=(2,2), activation='elu')(img_in)
        x = BatchNormalization()(x)
        x = Dropout(drop)(x)
        
        x = Convolution2D(36, (5,5), strides=(2,2), activation='elu')(x)
        x = BatchNormalization()(x)
        x = Dropout(drop)(x)
        
        x = Convolution2D(48, (5,5), strides=(2,2), activation='elu')(x)
        x = BatchNormalization()(x)
        x = Dropout(drop)(x)
        
        x = Convolution2D(64, (3,3), activation='elu')(x)
        x = BatchNormalization()(x)
        x = Dropout(drop)(x)
        
        x = Convolution2D(64, (3,3), activation='elu')(x)
        x = BatchNormalization()(x)
        x = Dropout(drop)(x)
        
        # 添加空间注意力模块
        attn = Convolution2D(1, (1, 1), activation='sigmoid')(x)
        x = Multiply()([x, attn])  # 特征图加权
        
        x = Flatten()(x)
        
        # 更深的全连接网络
        x = Dense(256, activation='elu')(x)
        x = Dropout(drop)(x)
        x = Dense(128, activation='elu')(x)
        x = Dropout(drop)(x)
        x = Dense(64, activation='elu')(x)
        x = Dropout(drop)(x)
        
        outputs = []
        for i in range(num_outputs):
            outputs.append(Dense(1, activation='linear', name='n_outputs'+str(i))(x))
            
        model = Model(inputs=[img_in], outputs=outputs, name='linear_enhanced')
        return model
    
    
    def run(self, img_arr: np.ndarray, *other_arr: list) -> Tuple[float, float]:
        '''数据预处理保留图片下半部分'''
        # 裁剪图像下半部分
        img_arr = img[img.shape[0]//2:, :, :]

        # 正规化（DonkeyCar 统一接口）
        norm_img_arr = normalize_image(img_arr)

        # 组织输入字典
        values = (norm_img_arr,) + tuple(np.array(arr) for arr in other_arr)
        input_dict = dict(zip(self.output_shapes()[0].keys(), values))

        # 推理并转换输出
        return self.inference_from_dict(input_dict)
    
 

    # 其余方法继承自KerasLinear保持不变
```



##  四、模型对比分析

|     对比维度     |        KerasLinear         |       KerasLinearEnhanced       |
| :--------------: | :------------------------: | :-----------------------------: |
|   **卷积结构**   | core_cnn_layers (固定结构) | 自定义CNN，层数更多，滤波器更多 |
|   **激活函数**   |            ReLU            |      ELU（更缓解梯度消失）      |
|    **正则化**    |       Dropout (0.2)        |    Dropout (0.3) + BatchNorm    |
|  **全连接结构**  |       Dense(100→50)        |     Dense(256→128→64)，更深     |
|   **损失函数**   |      MSE（全部输出）       | MSE(steering) + Huber(throttle) |
|   **损失权重**   |     默认均衡（无加权）     |  steering: 0.7, throttle: 0.3   |
|    **优化器**    |    可选 RMSProp / Adam     |          默认使用 Adam          |
| **输出范围限制** |             无             |               无                |
|   **算力要求**   |            较低            |              较高               |



## 五、模型效果展示

——在Donkeycar模拟器donkeycar simulator上和原模型同屏竞技

模拟器项目地址：[Releases · tawnkramer/gym-donkeycar](https://github.com/tawnkramer/gym-donkeycar/releases)

视频地址：[基于Donkeycar中KerasLinear模型的优化_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1uhEbz8E3w/?vd_source=ab275355fbf9a153dc896bb80b0dd8da)

<img src=".\image\Linear_VS_LinearEnhanced.jpg" alt="Linear_VS_LinearEnhanced" style="zoom:80%;" />



## 六、使用说明

官方文档提供了将自己构建的网络添加入Donkeycar训练的方法

文档地址：[donkeydocs/docs/dev_guide/model.md at master · autorope/donkeydocs](https://github.com/autorope/donkeydocs/blob/master/docs/dev_guide/model.md)

1. 将模型整个class添加到`donkeycar/parts/keras.py`中

2. 在`donkeycar/utils.py`的``函数中加入LinearEnhanced的选项

   ```python
   ...
   elif model_type == 'linear_enhanced':
       kl = KerasSensors(input_shape=input_shape)
   ...
   ```

3. 在开始训练时的命令应该为`donkey train --tub ./data --model ./models/pilot.h5 --type linear_enhanced`
