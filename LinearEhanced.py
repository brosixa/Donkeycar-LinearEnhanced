class KerasLinearEnhanced(KerasPilot):
    """
    增强版的KerasLinear模型
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


    def run(self, img_arr: np.ndarray, *other_arr: list) -> Tuple[float, float]:
        '''数据预处理保留图片下半部分'''
        # 裁剪图像下半部分
        img_arr = img[img.shape[0] // 2:, :, :]

        # 正规化（DonkeyCar 统一接口）
        norm_img_arr = normalize_image(img_arr)

        # 组织输入字典
        values = (norm_img_arr,) + tuple(np.array(arr) for arr in other_arr)
        input_dict = dict(zip(self.output_shapes()[0].keys(), values))

        # 推理并转换输出
        return self.inference_from_dict(input_dict)


    def create_model(self):
        return default_n_linear(self.num_outputs, self.input_shape)

    def compile(self):
        self.interpreter.compile(optimizer=self.optimizer, loss='mse')

    def interpreter_to_output(self, interpreter_out):
        steering = interpreter_out[0]
        throttle = interpreter_out[1]
        return steering[0], throttle[0]

    def y_transform(self, record: Union[TubRecord, List[TubRecord]]) \
            -> Dict[str, Union[float, List[float]]]:
        assert isinstance(record, TubRecord), 'TubRecord expected'
        angle: float = record.underlying['user/angle']
        throttle: float = record.underlying['user/throttle']
        return {'n_outputs0': angle, 'n_outputs1': throttle}

    def output_shapes(self):
        # need to cut off None from [None, 120, 160, 3] tensor shape
        img_shape = self.get_input_shape('img_in')[1:]
        shapes = ({'img_in': tf.TensorShape(img_shape)},
                  {'n_outputs0': tf.TensorShape([]),
                   'n_outputs1': tf.TensorShape([])})
        return shapes




def enhanced_linear_model(self, num_outputs, input_shape):
    """增强的线性模型架构"""
    drop = 0.3
    img_in = Input(shape=input_shape, name='img_in')

    # 使用更现代的CNN架构
    x = Convolution2D(24, (5, 5), strides=(2, 2), activation='elu')(img_in)
    x = BatchNormalization()(x)
    x = Dropout(drop)(x)

    x = Convolution2D(36, (5, 5), strides=(2, 2), activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dropout(drop)(x)

    x = Convolution2D(48, (5, 5), strides=(2, 2), activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dropout(drop)(x)

    x = Convolution2D(64, (3, 3), activation='elu')(x)
    x = BatchNormalization()(x)
    x = Dropout(drop)(x)

    x = Convolution2D(64, (3, 3), activation='elu')(x)
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
        outputs.append(Dense(1, activation='linear', name='n_outputs' + str(i))(x))

    model = Model(inputs=[img_in], outputs=outputs, name='linear_enhanced')
    return model