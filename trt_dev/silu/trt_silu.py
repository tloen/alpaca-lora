import numpy as np
from cuda import cudart
import torch
from torch import Tensor, nn
import tensorrt as trt

class SiLUActivation(nn.Module):
    """
    See Gaussian Error Linear Units (Hendrycks et al., https://arxiv.org/abs/1606.08415) where the SiLU (Sigmoid Linear
    Unit) was originally introduced and coined, and see Sigmoid-Weighted Linear Units for Neural Network Function
    Approximation in Reinforcement Learning (Elfwing et al., https://arxiv.org/abs/1702.03118) and Swish: a Self-Gated
    Activation Function (Ramachandran et al., https://arxiv.org/abs/1710.05941v1) where the SiLU was experimented with
    later.
    """

    def forward(self, input: Tensor) -> Tensor:
        return nn.functional.silu(input)
    
    def b_forward(self, input: Tensor) -> Tensor:
        return torch.matmul(input.T, nn.functional.sigmoid(input))

def test_trt(nIn, hIn, wIn, cOut, raw_data, weight, bias):
    data = np.array(raw_data)

    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)

    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()

    # input
    # second `1` is necessary for tensorRT, to turn this vector into NCHW
    inputT0 = network.add_input('inputT0', trt.DataType.FLOAT, (nIn, 1, hIn, -1))

    # dynamic shape optimization
    profile = builder.create_optimization_profile();
    profile.set_shape("inputT0", (nIn, 1, hIn, 1), (nIn, 1, hIn, 2), (nIn, 1, hIn, 3)) 
    config.add_optimization_profile(profile)

    # add fully connected layer
    selu_sigmoid_layer = network.add_activation(inputT0, type=trt.ActivationType.SIGMOID)
    selu_mult_layer = network.add_elementwise(inputT0, selu_sigmoid_layer.get_output(0), op=trt.ElementWiseOperation.PROD)

    # output
    network.mark_output(selu_mult_layer.get_output(0))

    engineString = builder.build_serialized_network(network, config)

    print("Runtime")
    engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
    context = engine.create_execution_context()

    # dynamic shape configure
    print("Set input shape")
    context.set_input_shape("inputT0", (nIn, 1, hIn, wIn))

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

    cudart.cudaStreamDestroy(stream)
    cudart.cudaFree(inputD0)
    cudart.cudaFree(outputD0)

    return outputH0

def test_torch(nIn, hIn, wIn, cOut, raw_data, weight, bias):
    # data = torch.tensor(raw_data).reshape(-1)
    # model = torch.nn.Linear(hIn * wIn, cOut)

    # # initialize model weights
    # model.weight.data.fill_(1)
    # print(model.weight.data.detach().cpu().numpy())
    # model.bias.data.fill_(0)
    model = SiLUActivation()

    output = model(data)

    return output


if __name__ == "__main__":
    # Input tensor shape NCHW
    nIn, hIn, wIn = 1, 2, 2

    # Output tensor shape C
    cOut = 2

    # Input tensor
    data = np.arange(hIn * wIn, dtype=np.float32).reshape(nIn, hIn, wIn)

    # fully connected weight
    weight = np.ones(cOut * hIn * wIn, dtype=np.float32).reshape(cOut, hIn * wIn)

    # fully connected bias
    bias = np.zeros(cOut, dtype=np.float32)
    
    print("inputH0 :", data.shape)
    print(data)
    
    output_trt = test_trt(nIn, hIn, wIn, cOut, data, weight, bias).reshape(-1)
    print("output_trt :", output_trt.shape)
    print(output_trt)

    output_torch = test_torch(nIn, hIn, wIn, cOut, data, weight, bias)
    print("output_torch :", output_torch.shape)
    print(output_torch)

    # assert(np.allclose(output_torch.detach().numpy(), output_trt))
