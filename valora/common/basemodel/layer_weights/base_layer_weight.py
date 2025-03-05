import torch
import numpy as np
import threading

class BaseLayerWeight:
    def __init__(self):
        self.tp_rank_ = None
        self.lock = threading.Lock()
        pass

    def load_hf_weights(self, weights):
        """
        load weights
        """
        pass


    def init_static_params(self):
        """
        design for some static init params, many model dont need do this.
        """
        pass

    def verify_load(self):
        """
        verify all load is ok
        """
        raise Exception("must verify weights load ok")
        pass

    def _cuda(self, cpu_tensor):
        return cpu_tensor.contiguous().to(self.data_type_).cuda()

    # def _try_cat_to(self, source_tensor_names, dest_name, cat_dim, handle_func=None):
    #     if all(hasattr(self, src_name) for src_name in source_tensor_names) and not hasattr(self, dest_name):
    #         with self.lock:
    #             if all(hasattr(self, src_name) for src_name in source_tensor_names) and not hasattr(self, dest_name):
    #                 # 打印每个张量的状态以调试
    #                 for name in source_tensor_names:
    #                     tensor = getattr(self, name, None)
    #                     if tensor is not None:
    #                         print(f"{name} is_cuda: {tensor.is_cuda}")

    #                 assert all(
    #                     not getattr(self, name, None).is_cuda for name in source_tensor_names
    #                 ), "all not cuda tensor"

    #                 tensors = [getattr(self, name, None) for name in source_tensor_names]
    #                 ans = torch.cat(tensors, dim=cat_dim)
    #                 if handle_func is not None:
    #                     ans = handle_func(ans)
    #                 else:
    #                     ans = self._cuda(ans)
    #                 setattr(self, dest_name, ans)
    #                 for name in source_tensor_names:
    #                     delattr(self, name)
    #     return
    
    def _try_cat_to(self, source_tensor_names, dest_name, cat_dim, handle_func=None):
        if all(hasattr(self, src_name) for src_name in source_tensor_names) and not hasattr(self, dest_name):
            with self.lock:
                if all(hasattr(self, src_name) for src_name in source_tensor_names) and not hasattr(self, dest_name):
                    # 打印每个张量的状态以调试
                    for name in source_tensor_names:
                        tensor = getattr(self, name, None)
                        if tensor is not None:
                            print(f"{name} is_cuda: {tensor.is_cuda}")

                    assert all(
                        not getattr(self, name, None).is_cuda for name in source_tensor_names
                    ), "all not cuda tensor"

                    tensors = [getattr(self, name, None) for name in source_tensor_names]
                    ans = torch.cat(tensors, dim=cat_dim)
                    if handle_func is not None:
                        ans = handle_func(ans)
                    else:
                        ans = self._cuda(ans)
                    setattr(self, dest_name, ans)
                    # for name in source_tensor_names:
                    #     delattr(self, name)
                    # 将原始张量移动到 CUDA 上
                    for name in source_tensor_names:
                        tensor = getattr(self, name, None)
                        if tensor is not None:
                            setattr(self, name, self._cuda(tensor))
        
        return