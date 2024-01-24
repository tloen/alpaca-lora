import numpy as np
from cuda import cudart
import torch
from torch import Tensor, nn
import tensorrt as trt

# RMSNorm by PyTorch
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # The gamma parameter
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        # (B, Seq_Len, Dim) * (B, Seq_Len, 1) = (B, Seq_Len, Dim)
        # rsqrt: 1 / sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        # (Dim) * (B, Seq_Len, Dim) = (B, Seq_Len, Dim)
        return self.weight * self._norm(x.float()).type_as(x)
    
def test_torch(nIn, hIn, wIn, cOut, raw_data):
    data = torch.tensor(raw_data).reshape(-1)
    
    model = RMSNorm(1)

    output = model(data)

    return output


def test_trt(nIn, hIn, wIn, cOut, raw_data):
    data = np.array(raw_data)

    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)

    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    
    # input
    inputT0 = network.add_input('inputT0', trt.DataType.FLOAT, [-1])
    avg_factor = np.array([nIn * hIn * wIn]).astype('float32')
    epsilon_weight = np.array([1e-06]).astype('float32')
    avg_tensor = network.add_constant(shape=list(avg_factor.shape), weights=trt.Weights(avg_factor))
    epsilon = network.add_constant(shape=list(epsilon_weight.shape), weights=trt.Weights(epsilon_weight))

    # dynamic shape optimization
    profile = builder.create_optimization_profile();
    profile.set_shape("inputT0", [1], [hIn*wIn], [nIn*hIn*wIn]) 
    config.add_optimization_profile(profile)
    
    # RMSNorm Layer: 1) Square: X^2 -> 2) Sum: sum of all x^2 -> 3) Mean: 1/N -> 4) Root: sqrt(X) -> 5) Division: 1/X
    # 1) Square: X^2
    RMSNorm_Square_layer = network.add_elementwise(inputT0, inputT0, op=trt.ElementWiseOperation.PROD)
    
    # 2) Sum: sum of all X^2
    RMSNorm_Sum_layer = network.add_reduce(RMSNorm_Square_layer.get_output(0), op=trt.ReduceOperation.SUM, axes=1, keep_dims=True)
    
    # 3) Mean: 1/N
    RMSNorm_Mean_layer = network.add_elementwise(RMSNorm_Sum_layer.get_output(0),
                                                 avg_tensor.get_output(0),
                                                 op=trt.ElementWiseOperation.DIV)
    
    # 4) Add epsilon
    RMSNorm_Mean_with_epsilon_layer = network.add_elementwise(RMSNorm_Mean_layer.get_output(0),
                                                              epsilon.get_output(0), op=trt.ElementWiseOperation.SUM)
    
    # 5) Root: sqrt(X)
    RMSNorm_Sqrt_layer = network.add_unary(RMSNorm_Mean_with_epsilon_layer.get_output(0), op=trt.UnaryOperation.SQRT)
    
    # 6) Division: 1/X
    RMSNorm_Div_layer = network.add_elementwise(inputT0, RMSNorm_Sqrt_layer.get_output(0), op=trt.ElementWiseOperation.DIV)
    
    # output
    network.mark_output(RMSNorm_Div_layer.get_output(0))

    engineString = builder.build_serialized_network(network, config)
    
    print("Runtime")
    engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)
    context = engine.create_execution_context()

    # dynamic shape configure
    print("Set input shape")
    context.set_input_shape("inputT0", [nIn * hIn * wIn])
    context.set_binding_shape(0, [nIn * hIn * wIn])

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
      
    
if __name__ == "__main__":
    # Input tensor shape NCHW
    nIn, hIn, wIn = 4, 45, 4096

    # Output tensor shape C
    cOut = 2

    # Input tensor
    data = np.arange(nIn * hIn * wIn, dtype=np.float32).reshape(nIn, hIn, wIn)

    # fully connected weight (not used)
    #weight = np.ones(cOut * hIn * wIn, dtype=np.float32).reshape(cOut, hIn * wIn)

    # fully connected bias (not used)
    #bias = np.zeros(cOut, dtype=np.float32)
    
    print("inputH0 :", data.shape)
    #print(data)
    
    output_trt = test_trt(nIn, hIn, wIn, cOut, data).reshape(-1)
    print("output_trt :", output_trt.shape)
    print(output_trt)

    output_torch = test_torch(nIn, hIn, wIn, cOut, data)
    print("output_torch :", output_torch.shape)
    print(output_torch)