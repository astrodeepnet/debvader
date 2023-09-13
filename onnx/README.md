# Set ups
```
# Optional
pip install onnxruntime
pip install -U tf2onnx
```

# Using ONNX to run the model
!!! Currently doesnt work as there are non supported Ops !!!


## Storing the model in ONNX format
In the deblending_dc2_images_2.ipynb, after loading the model

```
# save model for onnx
# net.save('../onnx/my_model')
import tf2onnx
import onnx

input_signature = [tf.TensorSpec([None, *input_shape], tf.float32, name='x')]
# Use from_function for tf functions
onnx_model, _ = tf2onnx.convert.from_keras(net, input_signature, opset=18)
onnx.save(onnx_model, "../onnx/net.onnx")
```
While the above runs, the model is not usable as there are non supported ops:
```shell
Tensorflow op [model_2/multivariate_normal_tri_l/tensor_coercible/value/MultivariateNormalTriL/sample/MultivariateNormalTriL/batch_shape_tensor/BroadcastArgs: BroadcastArgs] is not supported
Tensorflow op [model_2/multivariate_normal_tri_l/tensor_coercible/value/MultivariateNormalTriL/sample/MultivariateNormalTriL/batch_shape_tensor/BroadcastArgs_1: BroadcastArgs] is not supported
Tensorflow op [model_2/multivariate_normal_tri_l/MultivariateNormalTriL/MultivariateNormalTriL/shapes_from_loc_and_scale/BroadcastArgs: BroadcastArgs] is not supported
Tensorflow op [model_2/multivariate_normal_tri_l/tensor_coercible/value/MultivariateNormalTriL/sample/BatchBroadcastSampleNormal/sample/SampleNormal/sample/Normal/sample/BroadcastArgs: BroadcastArgs] is not supported
Tensorflow op [model_2/multivariate_normal_tri_l/tensor_coercible/value/MultivariateNormalTriL/sample/BatchBroadcastSampleNormal/sample/SampleNormal/sample/Normal/sample/BroadcastArgs_1: BroadcastArgs] is not supported
Tensorflow op [model_2/multivariate_normal_tri_l/tensor_coercible/value/MultivariateNormalTriL/sample/BatchBroadcastSampleNormal/sample/SampleNormal/batch_shape_tensor_1/Normal/batch_shape_tensor/BroadcastArgs: BroadcastArgs] is not supported
Tensorflow op [model_2/multivariate_normal_tri_l/tensor_coercible/value/MultivariateNormalTriL/sample/BatchBroadcastSampleNormal/sample/SampleNormal/batch_shape_tensor_1/Normal/batch_shape_tensor/BroadcastArgs_1: BroadcastArgs] is not supported
Tensorflow op [model_2/multivariate_normal_tri_l/tensor_coercible/value/MultivariateNormalTriL/sample/BatchBroadcastSampleNormal/sample/SampleNormal/batch_shape_tensor_1/BroadcastArgs: BroadcastArgs] is not supported
Tensorflow op [model_2/multivariate_normal_tri_l/tensor_coercible/value/MultivariateNormalTriL/sample/BatchBroadcastSampleNormal/sample/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/BroadcastArgs: BroadcastArgs] is not supported
Tensorflow op [model_2/multivariate_normal_tri_l/tensor_coercible/value/MultivariateNormalTriL/sample/BatchBroadcastSampleNormal/sample/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/BroadcastArgs_1: BroadcastArgs] is not supported
Tensorflow op [model_2/multivariate_normal_tri_l/tensor_coercible/value/MultivariateNormalTriL/sample/BatchBroadcastSampleNormal/sample/SampleNormal/batch_shape_tensor/BroadcastArgs: BroadcastArgs] is not supported
Tensorflow op [model_2/multivariate_normal_tri_l/tensor_coercible/value/MultivariateNormalTriL/sample/BatchBroadcastSampleNormal/sample/BatchBroadcastSampleNormal/batch_shape_tensor_1/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/BroadcastArgs: BroadcastArgs] is not supported
Tensorflow op [model_2/multivariate_normal_tri_l/tensor_coercible/value/MultivariateNormalTriL/sample/BatchBroadcastSampleNormal/sample/BatchBroadcastSampleNormal/batch_shape_tensor_1/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/BroadcastArgs_1: BroadcastArgs] is not supported
Tensorflow op [model_2/multivariate_normal_tri_l/tensor_coercible/value/MultivariateNormalTriL/sample/BatchBroadcastSampleNormal/sample/BatchBroadcastSampleNormal/batch_shape_tensor_1/SampleNormal/batch_shape_tensor/BroadcastArgs: BroadcastArgs] is not supported
Tensorflow op [model_2/multivariate_normal_tri_l/tensor_coercible/value/MultivariateNormalTriL/sample/BatchBroadcastSampleNormal/sample/BatchBroadcastSampleNormal/batch_shape_tensor_1/BroadcastArgs: BroadcastArgs] is not supported
Tensorflow op [model_2/multivariate_normal_tri_l/tensor_coercible/value/MultivariateNormalTriL/sample/BatchBroadcastSampleNormal/sample/BatchBroadcastSampleNormal/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/BroadcastArgs: BroadcastArgs] is not supported
Tensorflow op [model_2/multivariate_normal_tri_l/tensor_coercible/value/MultivariateNormalTriL/sample/BatchBroadcastSampleNormal/sample/BatchBroadcastSampleNormal/batch_shape_tensor/SampleNormal/batch_shape_tensor/Normal/batch_shape_tensor/BroadcastArgs_1: BroadcastArgs] is not supported
Tensorflow op [model_2/multivariate_normal_tri_l/tensor_coercible/value/MultivariateNormalTriL/sample/BatchBroadcastSampleNormal/sample/BatchBroadcastSampleNormal/batch_shape_tensor/SampleNormal/batch_shape_tensor/BroadcastArgs: BroadcastArgs] is not supported
Tensorflow op [model_2/multivariate_normal_tri_l/tensor_coercible/value/MultivariateNormalTriL/sample/BatchBroadcastSampleNormal/sample/BatchBroadcastSampleNormal/batch_shape_tensor/BroadcastArgs: BroadcastArgs] is not supported
Tensorflow op [model_2/model_1/distribution_lambda/tensor_coercible/value/Normal/sample/BroadcastArgs: BroadcastArgs] is not supported
Tensorflow op [model_2/model_1/distribution_lambda/tensor_coercible/value/Normal/sample/BroadcastArgs_1: BroadcastArgs] is not supported
Unsupported ops: Counter({'BroadcastArgs': 21, 'Log1p': 1})
```
## Running the model
```
import onnxruntime as ort
import numpy as np

# Change shapes and types to match model
input1 = np.zeros((20, *input_shape), np.float32)
results_tf = net(input1)

# Start from ORT 1.10, ORT requires explicitly setting the providers parameter if you want to use execution providers
# other than the default CPU provider (as opposed to the previous behavior of providers getting set/registered by default
# based on the build flags) when instantiating InferenceSession.
# Following code assumes NVIDIA GPU is available, you can specify other execution providers or don't include providers parameter
# to use default CPU provider.
sess = ort.InferenceSession(
    "../onnx/net.onnx", 
    # providers=["CUDAExecutionProvider"]
    )

# Set first argument of sess.run to None to use all model outputs in default order
# Input/output names are printed by the CLI and can be set with --rename-inputs and --rename-outputs
# If using the python API, names are determined from function arg names or TensorSpec names.
results_ort = sess.run(net.output_names, {"input1": input1})

for ort_res, tf_res in zip(results_ort, results_tf):
    np.testing.assert_allclose(ort_res, tf_res, rtol=1e-5, atol=1e-5)

print("Results match")
```

# What works: Storing just the encoder or decoder

```
import tf2onnx
import onnx

import onnxruntime as ort

input_signature = [tf.TensorSpec([None, *input_shape], tf.float32, name='x')]
# Use from_function for tf functions
onnx_model, _ = tf2onnx.convert.from_keras(encoder, input_signature, opset=18)
onnx.save(onnx_model, "../onnx/encoder.onnx")

# Change shapes and types to match model
input1 = np.zeros((20, *input_shape), np.float32)
results_tf = encoder(input1)

sess = ort.InferenceSession(
    "../onnx/encoder.onnx", 
    # providers=["CUDAExecutionProvider"]
    )

results_ort = sess.run(encoder.output_names, {"x": input1})
np.testing.assert_allclose(results_ort[0], results_tf, rtol=1e-5, atol=1e-5)
```