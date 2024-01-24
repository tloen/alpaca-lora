import numpy as np
from cuda import cudart
import torch
import tensorrt as trt

def test_trt(nIn, cIn, hIn, wIn, cOut, raw_data, weight, bias):
    data = np.array(raw_data)

    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)

    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()

    # input
    inputT0 = network.add_input('inputT0', trt.DataType.FLOAT, (nIn, cIn, hIn, -1))

    # dynamic shape optimization
    profile = builder.create_optimization_profile();
    profile.set_shape("inputT0", (nIn, cIn, hIn, 3), (nIn, cIn, hIn, 4), (nIn, cIn, hIn, 5)) 
    config.add_optimization_profile(profile)

    # add fully connected layer
    fullyConnectedLayer = network.add_fully_connected(inputT0, cOut, weight, bias)

    # output
    network.mark_output(fullyConnectedLayer.get_output(0))

    engineString = builder.build_serialized_network(network, config)

    print("Runtime")
    engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
    context = engine.create_execution_context()

    # dynamic shape configure
    print("Set input shape")
    context.set_input_shape("inputT0", (nIn, cIn, hIn, wIn))


    _, stream = cudart.cudaStreamCreate()

    inputH0 = np.ascontiguousarray(data.reshape(-1))
    outputH0 = np.empty(context.get_binding_shape(1), dtype=trt.nptype(engine.get_binding_dtype(1)))

    # initialize input and output data
    _, inputD0 = cudart.cudaMallocAsync(inputH0.nbytes, stream)
    _, outputD0 = cudart.cudaMallocAsync(outputH0.nbytes, stream)

    # move input to device
    cudart.cudaMemcpyAsync(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)

    # execute
    print("execute")
    context.execute_async_v2([int(inputD0), int(outputD0)], stream)

    # move output back to host
    cudart.cudaMemcpyAsync(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)

    # wait for everything
    cudart.cudaStreamSynchronize(stream)

    print("inputH0 :", data.shape)
    print(data)

    print("outputH0:", outputH0.shape)
    print(outputH0)

    cudart.cudaStreamDestroy(stream)
    cudart.cudaFree(inputD0)
    cudart.cudaFree(outputD0)

    return outputH0

def test_torch(nIn, cIn, hIn, wIn, cOut, raw_data, weight, bias):
    data = torch.tensor(raw_data).reshape(-1)
    model = torch.nn.Linear(cIn * hIn * wIn, cOut)

    # initialize model weights
    model.weight.data.fill_(1)
    model.bias.data.fill_(0)

    print(model.weight.shape)
    print(model.bias.shape)
    output = model(data)

    return output


if __name__ == "__main__":
    # Input tensor shape NCHW
    nIn, cIn, hIn, wIn = 1, 3, 4, 5

    # Output tensor shape C
    cOut = 2

    # Input tensor
    data = np.arange(cIn * hIn * wIn, dtype=np.float32).reshape(cIn, hIn, wIn)

    # fully connected weight
    weight = np.ones(cOut * cIn * hIn * wIn, dtype=np.float32).reshape(cOut, cIn, hIn, wIn)

    # fully connected bias
    bias = np.zeros(cOut, dtype=np.float32)
    
    test_trt(nIn, cIn, hIn, wIn, cOut, data, weight, bias)

    print(test_torch(nIn, cIn, hIn, wIn, cOut, data, weight, bias))
