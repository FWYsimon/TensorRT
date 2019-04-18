## TensorRT
### Preparation
1. TensorRT5.0+
2. cudnn7.0+

### How to use
The directory of generatorInt8 is used to generator int8 model.
You can generator classifier or detection model
To generate ssd or refinedet model, you may change your prototxt [follow this link](https://docs.nvidia.com/deeplearning/sdk/tensorrt-release-notes/tensorrt-5.html#rel_5-0-4).

The directory of tensorRTEngine is used to do inference.
You can test your int8 model(see the code in test.cpp)
