"""This is the file for the DRIVE network subclass"""

from network.retinal_w_masks import RetinalWMasksNetwork

class DriveNetwork(RetinalWMasksNetwork):

    # actual image dimensions 
    IMAGE_HEIGHT = 584
    IMAGE_WIDTH = 565

    # transformed input dimensions for network input
    FIT_IMAGE_HEIGHT = 584
    FIT_IMAGE_WIDTH = 584


    IMAGE_CHANNELS = 1
    
    def __init__(self, weight_init,learningrate,Beta1,Beta2,epsilon,regularizer=None,Relu=False,layers=None, skip_connections=True,**kwargs):
       # tf.reset_default_graph()
        self.regularizer=regularizer
        self.learningrate=learningrate
        self.Beta1=Beta1
        self.Beta2=Beta2
        self.epsilon=epsilon
        
        if layers == None:

            layers = []
            layers.append(Conv2d(kernel_size=3, output_channels=64, name='conv_1_1',Relu=Relu,weight_i=weight_init))
            layers.append(MaxPool2d(kernel_size=2, name='max_1', skip_connection=True and skip_connections))

            layers.append(Conv2d(kernel_size=3, output_channels=128, name='conv_2_1',Relu=Relu,weight_i=weight_init))

            layers.append(MaxPool2d(kernel_size=2, name='max_2', skip_connection=True and skip_connections))
            layers.append(Conv2d(kernel_size=3, output_channels=256, name='conv_3_1',Relu=Relu, weight_i=weight_init))
            layers.append(Conv2d(kernel_size=3, dilation=2, output_channels=256, name='conv_3_2',Relu=Relu,weight_i=weight_init))

            layers.append(MaxPool2d(kernel_size=2, name='max_3', skip_connection=True and skip_connections))

            layers.append(Conv2d(kernel_size=7, output_channels=4096, name='conv_4_1',Relu=Relu,weight_i=weight_init))
            layers.append(Conv2d(kernel_size=1, output_channels=4096, name='conv_4_2',Relu=Relu,weight_i=weight_init))

            self.inputs = tf.placeholder(tf.float32, [None, self.FIT_IMAGE_HEIGHT, self.FIT_IMAGE_WIDTH,
                                                      self.IMAGE_CHANNELS], name='inputs')
            self.masks = tf.placeholder(tf.float32, [None, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 1], name='masks')
            self.targets = tf.placeholder(tf.float32, [None, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 1], name='targets')
        super(DriveNetwork, self).__init__(layers=layers, **kwargs)

    def net_output(self, net):
       
        net = tf.image.resize_image_with_crop_or_pad(net, self.IMAGE_HEIGHT, self.IMAGE_WIDTH)
        net = tf.multiply(net, self.masks)
        self.segmentation_result = tf.sigmoid(net)
        self.targets = tf.multiply(self.targets, self.masks)
        print('segmentation_result.shape: {}, targets.shape: {}'.format(self.segmentation_result.get_shape(),self.targets.get_shape()))

        reg_term=0
        # get regulairzation terms
        if self.regularizer=='L2':
            regular_variable=tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            reg_term=tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(scale=0.1),regular_variable)
    

        self.cost_unweighted = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(self.targets, net, pos_weight=1))
        self.cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(self.targets, net,
                                                                            pos_weight=self.wce_pos_weight))+reg_term
        print('net.shape: {}'.format(net.get_shape()))
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learningrate,beta1=self.Beta1,beta2=self.Beta2,epsilon=self.epsilon).minimize(self.cost)
        with tf.name_scope('accuracy'):
            argmax_probs = tf.round(self.segmentation_result)  # 0x1
            correct_pred = tf.cast(tf.equal(argmax_probs, self.targets), tf.float32)
            self.accuracy = mask_op_and_mask_mean(correct_pred, self.masks, 1, self.IMAGE_HEIGHT, self.IMAGE_WIDTH)
            tf.summary.scalar('accuracy', self.accuracy)

        self.summaries = tf.summary.merge_all()

