import numpy as np
from cuda import cudart
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor, nn
import tensorrt as trt
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

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
        return input * nn.functional.sigmoid(input)

class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config['hidden_size']
        self.intermediate_size = config['intermediate_size']
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = SiLUActivation()
        self.init = False

    def export(self):
        batch_size = 4
        # self.gate_proj.weight
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.builder = trt.Builder(self.logger)
        self.network = self.builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        self.config = self.builder.create_builder_config()

        # input
        # second `1` is necessary for tensorRT, to turn this vector into NCHW
        inputT0 = self.network.add_input('inputT0', trt.DataType.FLOAT, (batch_size, 1, -1, 4096))

        # dynamic shape optimization
        profile = self.builder.create_optimization_profile()
        profile.set_shape("inputT0", (batch_size, 1, 1, 4096), (batch_size, 1, 1, 4096), (batch_size, 1, 256, 4096))
        self.config.add_optimization_profile(profile)

        # self.up_proj(x)
        up_proj_weight = torch.tensor(self.up_proj.weight).cpu().numpy()
        up_proj_layer = self.network.add_fully_connected(inputT0, self.intermediate_size, up_proj_weight)

        # act_fn(self.gate_proj(x))
        gate_proj_weight = torch.tensor(self.gate_proj.weight).cpu().numpy()
        gate_proj_layer = self.network.add_fully_connected(inputT0, self.intermediate_size, gate_proj_weight)

        selu_sigmoid_layer = self.network.add_activation(gate_proj_layer.get_output(0), type=trt.ActivationType.SIGMOID)
        selu_mult_layer = self.network.add_elementwise(gate_proj_layer.get_output(0), selu_sigmoid_layer.get_output(0), op=trt.ElementWiseOperation.PROD)

        # act_fn(self.gate_proj(x)) * self.up_proj(x)
        before_down_proj_layer = self.network.add_elementwise(selu_mult_layer.get_output(0), up_proj_layer.get_output(0), op=trt.ElementWiseOperation.PROD)

        down_proj_weight = torch.tensor(self.down_proj.weight).cpu().numpy()
        down_proj_layer = self.network.add_fully_connected(before_down_proj_layer.get_output(0), self.hidden_size, down_proj_weight)

        # output
        self.network.mark_output(down_proj_layer.get_output(0))

        self.engineString = self.builder.build_serialized_network(self.network, self.config)

        self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(self.engineString)
        self.context = self.engine.create_execution_context()

        print("Completed creating Engine")
        with open("trt_mlp.trt", "wb") as f:
            f.write(self.engine.serialize())

    def load(self, dir):
        weights = torch.load(dir)
        mlp_weights = dict()
        for key in weights.keys():
            if key.split(".")[3] == "mlp":
                mlp_weights[key[key.find(key.split(".")[4]):]] = weights[key]

        self.load_state_dict(mlp_weights)

    def trt_load(self, dir):
        batch_size = 4
        # self.gate_proj.weight
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.builder = trt.Builder(self.logger)
        self.network = self.builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        self.config = self.builder.create_builder_config()
        with open(dir, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj

    def trt_forward(self, x):
        # dynamic shape configure
        print("Set input shape")
        self.context.set_input_shape("inputT0", (4, 1, 1, 4096))

        _, stream = cudart.cudaStreamCreate()

        inputH0 = np.ascontiguousarray(x.reshape(-1))
        outputH0 = np.empty(self.context.get_binding_shape(1), dtype=trt.nptype(self.engine.get_binding_dtype(1)))

        # initialize input and output data
        _, inputD0 = cudart.cudaMallocAsync(inputH0.nbytes, stream)
        _, outputD0 = cudart.cudaMallocAsync(outputH0.nbytes, stream)

        # move input to device
        cudart.cudaMemcpyAsync(inputD0, inputH0.ctypes.data, inputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)

        # execute
        print("execute")
        self.context.execute_async_v2([int(inputD0), int(outputD0)], stream)

        # move output back to host
        cudart.cudaMemcpyAsync(outputH0.ctypes.data, outputD0, outputH0.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)

        # wait for everything
        cudart.cudaStreamSynchronize(stream)

        cudart.cudaStreamDestroy(stream)
        cudart.cudaFree(inputD0)
        cudart.cudaFree(outputD0)

        # return down_proj
        return outputH0


if __name__ == "__main__":
    # activation tester
    # act_fn = SiLUActivation()
    
    # input = torch.ones(4, 2)
    # print(act_fn(input))
    # print(act_fn.b_forward(input))

    config = dict()
    config['hidden_size'] = 4096
    config['intermediate_size'] = 11008
    model = LlamaMLP(config)


    model.load("/home/fuchiang137/.cache/huggingface/hub/models--decapoda-research--llama-7b-hf/snapshots/5f98eefcc80e437ef68d457ad7bf167c2c6a1348/pytorch_model-00019-of-00033.bin")

    input = torch.ones(4, 1, 4096)
    output = model(input)
    print(output)
    print(output.shape)
    # model.export()

    model.trt_load("/home/fuchiang137/LLM_infer/trt_LLM/alpaca-lora/trt_mlp.trt")
    input = np.ones((4, 1, 4096))
    output_trt = model.trt_forward(input)
    output_trt = output_trt.reshape(output.shape)
    print("output_trt :", output_trt.shape)
    print(output_trt)
