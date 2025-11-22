"""
ResNet-18 model implementation for crop yield prediction.
This implementation is based on TensorFlow 1.15 and is compatible with the existing training loop.
"""

import tensorflow as tf
from models.base_model import BaseModel

# Constants for ResNet implementation
BN_DECAY = 0.99
CONV_WEIGHT_STDDEV = 0.1
FC_WEIGHT_STDDEV = 0.01
DEFAULT_DTYPE = tf.float32


class ResNet18(BaseModel):
    """
    ResNet-18 model for crop yield prediction.
    
    This implementation follows the ResNet v2 (pre-activation) architecture
    and is optimized for satellite image data.
    """
    
    def __init__(self, inputs, num_outputs=1, is_training=True,
                 fc_reg=0.0003, conv_reg=0.0003,
                 use_dilated_conv_in_first_layer=False):
        """
        Args:
        - inputs: tf.Tensor, shape [batch_size, H, W, C], type float32
        - num_outputs: int, number of output classes (default 1 for regression)
        - is_training: bool, or tf.placeholder of type tf.bool
        - fc_reg: float, regularization for weights in fully-connected layer
        - conv_reg: float, regularization for weights in conv layers
        - use_dilated_conv_in_first_layer: bool
        """
        super(ResNet18, self).__init__(
            inputs=inputs,
            num_outputs=num_outputs,
            is_training=is_training,
            fc_reg=fc_reg,
            conv_reg=conv_reg)
        
        # ResNet-18 specific configuration
        self.num_blocks = [2, 2, 2, 2]  # ResNet-18 has 2 blocks in each of the 4 groups
        self.bottleneck = False  # ResNet-18 uses basic blocks, not bottleneck blocks
        
        # Build the network
        self.outputs, self.features_layer = self._build_resnet(
            inputs,
            is_training=is_training,
            num_classes=num_outputs,
            num_blocks=self.num_blocks,
            bottleneck=self.bottleneck,
            use_dilated_conv_in_first_layer=use_dilated_conv_in_first_layer,
            conv_reg=conv_reg,
            fc_reg=fc_reg
        )
    
    def _build_resnet(self, x, is_training, num_classes, num_blocks, bottleneck,
                     use_dilated_conv_in_first_layer, conv_reg, fc_reg):
        """
        Build the ResNet-18 architecture.
        
        Args:
        - x: tf.Tensor, shape [batch_size, H, W, C], type float32
        - is_training: bool
        - num_classes: int, number of output classes
        - num_blocks: list of 4 integers, number of blocks in each of the 4 groups
        - bottleneck: bool, if True uses bottleneck layer (False for ResNet-18)
        - use_dilated_conv_in_first_layer: bool
        - conv_reg: float, L2 weight regularization penalty for conv layers
        - fc_reg: float, L2 weight regularization penalty for fully-connected layer
        
        Returns:
        - x: tf.Tensor with shape [batch_size, num_classes]
        - features_layer: tf.Tensor with shape [batch_size, 512]
        """
        with tf.variable_scope('resnet18'):
            # Initial convolution
            with tf.variable_scope('scale1'):
                x = self._conv(x, 64, kernel_size=7, stride=2, reg=conv_reg, name='conv1')
                x = self._batch_norm_activation(x, is_training, name='bn1')
            
            # Max pooling
            x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
            
            # Residual blocks
            # Scale 2: 64 filters
            with tf.variable_scope('scale2'):
                x = self._residual_block(x, 64, 64, is_training, stride=1, reg=conv_reg, name='block1')
                x = self._residual_block(x, 64, 64, is_training, stride=1, reg=conv_reg, name='block2')
            
            # Scale 3: 128 filters
            with tf.variable_scope('scale3'):
                x = self._residual_block(x, 64, 128, is_training, stride=2, reg=conv_reg, name='block1')
                x = self._residual_block(x, 128, 128, is_training, stride=1, reg=conv_reg, name='block2')
            
            # Scale 4: 256 filters
            with tf.variable_scope('scale4'):
                x = self._residual_block(x, 128, 256, is_training, stride=2, reg=conv_reg, name='block1')
                x = self._residual_block(x, 256, 256, is_training, stride=1, reg=conv_reg, name='block2')
            
            # Scale 5: 512 filters
            with tf.variable_scope('scale5'):
                x = self._residual_block(x, 256, 512, is_training, stride=2, reg=conv_reg, name='block1')
                x = self._residual_block(x, 512, 512, is_training, stride=1, reg=conv_reg, name='block2')
            
            # Global average pooling
            features_layer = tf.reduce_mean(x, axis=[1, 2], name='global_avg_pool')
            
            # Fully connected layer for classification/regression
            if num_classes is not None:
                with tf.variable_scope('fc'):
                    x = self._fully_connected(features_layer, num_classes, reg=fc_reg, name='fc1')
            
            return x, features_layer
    
    def _residual_block(self, x, in_filters, out_filters, is_training, stride=1, reg=0.0, name='block'):
        """
        Basic residual block for ResNet-18.
        
        Args:
        - x: Input tensor
        - in_filters: Number of input filters
        - out_filters: Number of output filters
        - is_training: Whether in training mode
        - stride: Stride for the first convolution
        - reg: L2 regularization factor
        - name: Name scope for the block
        
        Returns:
        - Output tensor
        """
        with tf.variable_scope(name):
            # Shortcut connection
            shortcut = x
            if in_filters != out_filters or stride != 1:
                shortcut = self._conv(shortcut, out_filters, kernel_size=1, stride=stride, reg=reg, name='shortcut')
                shortcut = self._batch_norm(shortcut, is_training, name='shortcut_bn')
            
            # First convolution
            x = self._batch_norm(x, is_training, name='bn1')
            x = tf.nn.relu(x)
            x = self._conv(x, out_filters, kernel_size=3, stride=stride, reg=reg, name='conv1')
            
            # Second convolution
            x = self._batch_norm(x, is_training, name='bn2')
            x = tf.nn.relu(x)
            x = self._conv(x, out_filters, kernel_size=3, stride=1, reg=reg, name='conv2')
            
            # Add shortcut
            x = x + shortcut
            
            return x
    
    def _conv(self, x, filters, kernel_size=3, stride=1, reg=0.0, name='conv'):
        """
        Convolutional layer with L2 regularization.
        
        Args:
        - x: Input tensor
        - filters: Number of output filters
        - kernel_size: Size of the convolution kernel
        - stride: Stride for the convolution
        - reg: L2 regularization factor
        - name: Name of the layer
        
        Returns:
        - Output tensor
        """
        with tf.variable_scope(name):
            in_channels = x.get_shape()[-1]
            shape = [kernel_size, kernel_size, in_channels, filters]
            
            # Weight initialization
            initializer = tf.variance_scaling_initializer(scale=2.0, mode='fan_out', distribution='normal')
            
            weights = tf.get_variable(
                'weights',
                shape=shape,
                initializer=initializer,
                regularizer=tf.contrib.layers.l2_regularizer(reg) if reg > 0 else None
            )
            
            return tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')
    
    def _batch_norm(self, x, is_training, name='bn'):
        """
        Batch normalization layer.
        
        Args:
        - x: Input tensor
        - is_training: Whether in training mode
        - name: Name of the layer
        
        Returns:
        - Output tensor
        """
        with tf.variable_scope(name):
            return tf.layers.batch_normalization(
                x,
                momentum=BN_DECAY,
                training=is_training
            )
    
    def _batch_norm_activation(self, x, is_training, name='bn_relu'):
        """
        Batch normalization followed by ReLU activation.
        
        Args:
        - x: Input tensor
        - is_training: Whether in training mode
        - name: Name of the layer
        
        Returns:
        - Output tensor
        """
        with tf.variable_scope(name):
            x = self._batch_norm(x, is_training, name='bn')
            x = tf.nn.relu(x)
            return x
    
    def _fully_connected(self, x, units, reg=0.0, name='fc'):
        """
        Fully connected layer with L2 regularization.
        
        Args:
        - x: Input tensor
        - units: Number of output units
        - reg: L2 regularization factor
        - name: Name of the layer
        
        Returns:
        - Output tensor
        """
        with tf.variable_scope(name):
            in_units = x.get_shape()[-1]
            
            weights = tf.get_variable(
                'weights',
                shape=[in_units, units],
                initializer=tf.truncated_normal_initializer(stddev=FC_WEIGHT_STDDEV),
                regularizer=tf.contrib.layers.l2_regularizer(reg) if reg > 0 else None
            )
            
            biases = tf.get_variable(
                'biases',
                shape=[units],
                initializer=tf.zeros_initializer()
            )
            
            return tf.nn.xw_plus_b(x, weights, biases)
    
    def get_first_layer_weights(self):
        """
        Gets the weights in the first layer of the CNN.
        
        Returns:
        - tf.Tensor, shape [F_height, F_width, F_channels, num_filters]
        """
        with tf.variable_scope('resnet18/scale1', reuse=True):
            return tf.get_variable('conv1/weights')
    
    def get_final_layer_weights(self):
        """
        Gets the weights in the final fully-connected layer after the conv layers.
        
        Returns:
        - list of tf.Tensor
        """
        return tf.trainable_variables(scope='resnet18/fc')
    
    def get_first_layer_summaries(self, ls_bands=None, nl_band=None):
        """
        Creates summaries for the first layer weights.
        
        Args:
        - ls_bands: one of [None, 'rgb', 'ms']
        - nl_band: one of [None, 'split', 'merge']
        
        Returns:
        - summaries: tf.summary, merged summaries
        """
        summaries = []
        
        # Get first layer weights
        weights = self.get_first_layer_weights()
        weights_hist = tf.summary.histogram('first_layer_weights', weights)
        summaries.append(weights_hist)
        
        # Add band-specific histograms if needed
        if ls_bands in ['rgb', 'ms']:
            weights_rgb_hist = tf.summary.histogram('first_layer_weights_RGB', weights[:, :, 0:3, :])
            summaries.append(weights_rgb_hist)
        
        if ls_bands == 'ms':
            weights_ms_hist = tf.summary.histogram('first_layer_weights_MS', weights[:, :, 3:7, :])
            summaries.append(weights_ms_hist)
        
        if nl_band == 'merge':
            weights_nl_hist = tf.summary.histogram('first_layer_weights_NL', weights[:, :, -1:, :])
            summaries.append(weights_nl_hist)
        elif nl_band == 'split':
            weights_nl_hist = tf.summary.histogram('first_layer_weights_NL', weights[:, :, -2:, :])
            summaries.append(weights_nl_hist)
        
        return tf.summary.merge(summaries) if summaries else None
    
    def init_from_numpy(self, path, sess, hs_weight_init='random'):
        """
        Initialize weights from a numpy file (e.g., ImageNet pre-trained weights).
        
        Args:
        - path: str, path to a .npz file of pre-trained weights
        - sess: tf.Session
        - hs_weight_init: str, method for initializing weights of non-RGB bands
        """
        # This method can be implemented later if needed for transfer learning
        pass