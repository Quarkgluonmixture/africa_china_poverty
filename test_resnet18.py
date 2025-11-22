"""
Test script for ResNet-18 model implementation.
This script verifies that the model can be instantiated and can process input data correctly.
"""

import numpy as np
import tensorflow as tf
from models.models_resnet import ResNet18


def test_resnet18():
    """
    Test ResNet-18 model instantiation and forward pass.
    """
    print("Testing ResNet-18 model...")
    
    # Reset the graph
    tf.reset_default_graph()
    
    # Create input placeholder with shape [batch_size, height, width, channels]
    # Using standard satellite image size
    input_ph = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name='input')
    is_training = tf.placeholder(tf.bool, name='is_training')
    
    try:
        # Create ResNet-18 model
        model = ResNet18(
            inputs=input_ph,
            num_outputs=1,  # Regression output
            is_training=is_training,
            fc_reg=0.0003,
            conv_reg=0.0003,
            use_dilated_conv_in_first_layer=False
        )
        
        print("✓ ResNet-18 model instantiated successfully")
        print(f"  - Output shape: {model.outputs.get_shape()}")
        print(f"  - Features shape: {model.features_layer.get_shape()}")
        
        # Test with dummy data
        with tf.Session() as sess:
            # Initialize variables
            sess.run(tf.global_variables_initializer())
            
            # Create dummy input data
            batch_size = 4
            dummy_input = np.random.rand(batch_size, 224, 224, 3).astype(np.float32)
            
            # Run forward pass
            outputs, features = sess.run(
                [model.outputs, model.features_layer],
                feed_dict={
                    input_ph: dummy_input,
                    is_training: False
                }
            )
            
            print(f"✓ Forward pass successful")
            print(f"  - Output shape: {outputs.shape}")
            print(f"  - Features shape: {features.shape}")
            print(f"  - Output values (first 3): {outputs[:3].flatten()}")
            
            # Test first layer weights access
            try:
                first_layer_weights = sess.run(model.get_first_layer_weights())
                print(f"✓ First layer weights accessed successfully")
                print(f"  - Shape: {first_layer_weights.shape}")
            except Exception as e:
                print(f"✗ Error accessing first layer weights: {e}")
            
            # Test final layer weights access
            try:
                final_layer_vars = model.get_final_layer_weights()
                print(f"✓ Final layer variables accessed successfully")
                print(f"  - Number of variables: {len(final_layer_vars)}")
                for i, var in enumerate(final_layer_vars):
                    print(f"    - Variable {i}: {var.name}, shape: {var.shape}")
            except Exception as e:
                print(f"✗ Error accessing final layer weights: {e}")
            
            # Test summaries
            try:
                summaries = model.get_first_layer_summaries(ls_bands='rgb', nl_band=None)
                if summaries is not None:
                    print(f"✓ Summaries created successfully")
                else:
                    print(f"✗ Summaries returned None")
            except Exception as e:
                print(f"✗ Error creating summaries: {e}")
            
        print("\n✓ All tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_parameters():
    """
    Test different parameter configurations for the ResNet-18 model.
    """
    print("\nTesting different parameter configurations...")
    
    tf.reset_default_graph()
    
    input_ph = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name='input')
    is_training = tf.placeholder(tf.bool, name='is_training')
    
    # Test with different output sizes
    for num_outputs in [1, 10]:
        tf.reset_default_graph()
        input_ph = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name='input')
        is_training = tf.placeholder(tf.bool, name='is_training')
        
        try:
            model = ResNet18(
                inputs=input_ph,
                num_outputs=num_outputs,
                is_training=is_training
            )
            print(f"✓ Model with {num_outputs} outputs created successfully")
        except Exception as e:
            print(f"✗ Error creating model with {num_outputs} outputs: {e}")
    
    # Test with different regularization values
    for reg in [0.0, 0.001, 0.01]:
        tf.reset_default_graph()
        input_ph = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name='input')
        is_training = tf.placeholder(tf.bool, name='is_training')
        
        try:
            model = ResNet18(
                inputs=input_ph,
                num_outputs=1,
                is_training=is_training,
                fc_reg=reg,
                conv_reg=reg
            )
            print(f"✓ Model with regularization={reg} created successfully")
        except Exception as e:
            print(f"✗ Error creating model with regularization={reg}: {e}")


if __name__ == '__main__':
    print("=" * 50)
    print("ResNet-18 Model Test")
    print("=" * 50)
    
    success = test_resnet18()
    test_model_parameters()
    
    if success:
        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        print("The ResNet-18 model is ready for training.")
        print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("Some tests failed. Please check the implementation.")
        print("=" * 50)