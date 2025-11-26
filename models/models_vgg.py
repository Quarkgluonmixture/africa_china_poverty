"""
VGG-F model implementation for poverty prediction baseline.
This implementation follows the architecture from Jean et al. (2016) paper.
Based on TensorFlow 1.15 for compatibility with existing training infrastructure.
"""

import tensorflow as tf
from models.base_model import BaseModel

# Constants for VGG-F implementation
BN_DECAY = 0.9
CONV_WEIGHT_STDDEV = 0.1
FC_WEIGHT_STDDEV = 0.01
DEFAULT_DTYPE = tf.float32


class VGGF(BaseModel):
    """
    VGG-F model for poverty prediction baseline.
    
    Architecture follows Jean et al. (2016):
    - Conv1: 64 filters, 11x11, stride 4, ReLU
    - MaxPool: 3x3, stride 2
    - Conv2: 256 filters, 5x5, padding='SAME', ReLU
    - MaxPool: 3x3, stride 2
    - Conv3: 256 filters, 3x3, padding='SAME', ReLU
    - Conv4: 256 filters, 3x3, padding='SAME', ReLU
    - Conv5: 256 filters, 3x3, padding='SAME', ReLU
    - MaxPool: 3x3, stride 2
    - FC6: 4096 units, ReLU, Dropout (0.5)
    - FC7: 4096 units, ReLU, Dropout (0.5)
    - FC8 (Output): num_outputs units (1 for regression), no activation
    """
    
    def __init__(self, inputs, num_outputs=1, is_training=True,
                 fc_reg=0.0003, conv_reg=0.0003):
        """
        Args:
        - inputs: tf.Tensor, shape [batch_size, 224, 224, 3], type float32
        - num_outputs: int, number of output units (default 1 for regression)
        - is_training: bool, or tf.placeholder of type tf.bool
        - fc_reg: float, regularization for weights in fully-connected layers
        - conv_reg: float, regularization for weights in conv layers
        """
        super(VGGF, self).__init__(
            inputs=inputs,
            num_outputs=num_outputs,
            is_training=is_training,
            fc_reg=fc_reg,
            conv_reg=conv_reg)
        
        # Build the VGG-F network
        self.outputs, self.features_layer = self._build_vggf(
            inputs,
            is_training=is_training,
            num_outputs=num_outputs,
            conv_reg=conv_reg,
            fc_reg=fc_reg
        )

    
    def _build_vggf(self, x, is_training, num_outputs, conv_reg, fc_reg):
        """
        Build the VGG-F architecture.
        
        Args:
        - x: tf.Tensor, shape [batch_size, 224, 224, 3], type float32
        - is_training: bool or tf.placeholder
        - num_outputs: int, number of output units
        - conv_reg: float, L2 weight regularization penalty for conv layers
        - fc_reg: float, L2 weight regularization penalty for fully-connected layers
        
        Returns:
        - outputs: tf.Tensor with shape [batch_size, num_outputs]
        - features_layer: tf.Tensor with shape [batch_size, 4096] (FC7 output)
        """
        with tf.variable_scope('vggf'):
            # Conv1: 64 filters, 11x11, stride 4, ReLU
            with tf.variable_scope('conv1'):
                x = self._conv(x, filters=64, kernel_size=11, stride=4, reg=conv_reg)
                x = tf.nn.relu(x)
            
            # MaxPool1: 3x3, stride 2
            x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
            
            # Conv2: 256 filters, 5x5, padding='SAME', ReLU
            with tf.variable_scope('conv2'):
                x = self._conv(x, filters=256, kernel_size=5, stride=1, reg=conv_reg)
                x = tf.nn.relu(x)
            
            # MaxPool2: 3x3, stride 2
            x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
            
            # Conv3: 256 filters, 3x3, padding='SAME', ReLU
            with tf.variable_scope('conv3'):
                x = self._conv(x, filters=256, kernel_size=3, stride=1, reg=conv_reg)
                x = tf.nn.relu(x)
            
            # Conv4: 256 filters, 3x3, padding='SAME', ReLU
            with tf.variable_scope('conv4'):
                x = self._conv(x, filters=256, kernel_size=3, stride=1, reg=conv_reg)
                x = tf.nn.relu(x)
            
            # Conv5: 256 filters, 3x3, padding='SAME', ReLU
            with tf.variable_scope('conv5'):
                x = self._conv(x, filters=256, kernel_size=3, stride=1, reg=conv_reg)
                x = tf.nn.relu(x)
            
            # MaxPool3: 3x3, stride 2
            x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
            
            # Flatten for fully connected layers
            x = tf.layers.flatten(x)
            
            # FC6: 4096 units, ReLU, Dropout (0.5)
            with tf.variable_scope('fc6'):
                x = self._fully_connected(x, units=4096, reg=fc_reg)
                x = tf.nn.relu(x)
                x = tf.layers.dropout(x, rate=0.5, training=is_training)
            
            # FC7: 4096 units, ReLU, Dropout (0.5)
            with tf.variable_scope('fc7'):
                x = self._fully_connected(x, units=4096, reg=fc_reg)
                x = tf.nn.relu(x)
                x = tf.layers.dropout(x, rate=0.5, training=is_training)
            
            # Store FC7 output as features layer
            features_layer = x
            
            # FC8 (Output): num_outputs units, no activation
            if num_outputs is not None:
                with tf.variable_scope('fc8'):
                    x = self._fully_connected(x, units=num_outputs, reg=fc_reg)
            
            return x, features_layer
    
    def _conv(self, x, filters, kernel_size=3, stride=1, reg=0.0):
        """
        Convolutional layer with L2 regularization.
        
        Args:
        - x: Input tensor
        - filters: Number of output filters
        - kernel_size: Size of the convolution kernel
        - stride: Stride for the convolution
        - reg: L2 regularization factor
        
        Returns:
        - Output tensor
        """
        in_channels = x.get_shape()[-1]
        shape = [kernel_size, kernel_size, in_channels, filters]
        
        # Weight initialization
        initializer = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)
        
        weights = tf.get_variable(
            'weights',
            shape=shape,
            initializer=initializer,
            regularizer=tf.contrib.layers.l2_regularizer(reg) if reg > 0 else None
        )
        
        biases = tf.get_variable(
            'biases',
            shape=[filters],
            initializer=tf.zeros_initializer()
        )
        
        conv = tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')
        return tf.nn.bias_add(conv, biases)
    
    def _fully_connected(self, x, units, reg=0.0):
        """
        Fully connected layer with L2 regularization.
        
        Args:
        - x: Input tensor
        - units: Number of output units
        - reg: L2 regularization factor
        
        Returns:
        - Output tensor
        """
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
        with tf.variable_scope('vggf/conv1', reuse=True):
            return tf.get_variable('weights')
    
    def get_final_layer_weights(self):
        """
        Gets the weights in the final fully-connected layer after the conv layers.
        
        Returns:
        - list of tf.Tensor
        """
        return tf.trainable_variables(scope='vggf/fc8')
    
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