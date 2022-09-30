import imp
from .graph.graph import Graph
from .zig.zig import automated_partition_zigs
from .optimizer import DHSPG
from .compression.compression import automated_compression
import os
from .flops.flops import compute_flops

class OTO:
    def __init__(self, model=None, dummy_input=None):
        self._graph = None
        self._model = model
        self._dummy_input = dummy_input

        if self._model is not None and self._dummy_input is not None:
            self.initialize(model=self._model, dummy_input=self._dummy_input)
            self.partition_zigs()

    def initialize(self, model=None, dummy_input=None):
        model = model.eval()
        self._model = model
        self._dummy_input = dummy_input
        self._graph = Graph(model, dummy_input)

    def partition_zigs(self):
        self._graph = automated_partition_zigs(self._graph)
    
    def visulize_zigs(self, out_dir=None, view=True):
        self._graph.build_dot(verbose=True).render(\
            os.path.join(out_dir if out_dir is not None else './', \
                self._model.name if hasattr(self._model, 'name') else type(self._model).__name__ + '_zig.gv'), \
                view=view)

    def dhspg(self, lr=0.1, lmbda=0.0, lmbda_amplify=1.1, hat_lmbda_coeff=10, weight_decay=0.0, first_momentum=0.0, second_momentum=0.0, \
               adam=False, target_group_sparsity=0.5, tolerance_group_sparsity=0.01, partition_step=1e5, half_space_project_steps=2e4,\
               warm_up_steps=None, dampening=0.0, epsilon=[]):
        self._optimizer = DHSPG(
            params=self._graph.params_groups(epsilon=epsilon),
            lr=lr,
            lmbda=lmbda,
            lmbda_amplify=lmbda_amplify,
            hat_lmbda_coeff=hat_lmbda_coeff,
            weight_decay=weight_decay,
            first_momentum=first_momentum,
            second_momentum=second_momentum,
            dampening=dampening,
            adam=adam,
            target_group_sparsity=target_group_sparsity, 
            tolerance_group_sparsity=tolerance_group_sparsity,
            partition_step=partition_step,
            warm_up_steps=warm_up_steps,
            half_space_project_steps=half_space_project_steps)
        return self._optimizer

    def compress(self, compressed_model_path=None, dynamic_axes=False):
        _, self.compressed_model_path = automated_compression(
            oto_graph=self._graph,
            model=self._model,
            dummy_input=self._dummy_input,
            compressed_model_path=compressed_model_path,
            dynamic_axes=dynamic_axes)
    
    def random_set_zero_groups(self):
        self._graph.random_set_zero_groups()
    
    def compute_flops(self, compressed=False):
        return compute_flops(self._graph, compressed=compressed)

    def compute_num_params(self):
        return sum([w.numel() for name, w in self._model.named_parameters()])

    def compute_num_params(self, compressed=False):
        if not compressed:
            return sum([w.numel() for w in self._model.state_dict().values()])   
        else:
            # load compressed model
            import onnx
            import numpy as np
            compressed_onnx = onnx.load(self.compressed_model_path)
            compressed_onnx_graph = compressed_onnx.graph
            num_params_compressed = 0
            for tensor in compressed_onnx_graph.initializer:
                num_params_compressed += np.prod(tensor.dims)
            return num_params_compressed
