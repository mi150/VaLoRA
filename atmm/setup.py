import pathlib
from setuptools import setup, find_packages
import torch.utils.cpp_extension as torch_cpp_ext
import os
root = pathlib.Path(__name__).parent



def glob(pattern):
  return [str(p) for p in root.glob(pattern)]

def remove_unwanted_pytorch_nvcc_flags():
  REMOVE_NVCC_FLAGS = [
      '-D__CUDA_NO_HALF_OPERATORS__',
      '-D__CUDA_NO_HALF_CONVERSIONS__',
      '-D__CUDA_NO_BFLOAT16_CONVERSIONS__',
      '-D__CUDA_NO_HALF2_OPERATORS__',
  ]
  for flag in REMOVE_NVCC_FLAGS:
    try:
      torch_cpp_ext.COMMON_NVCC_FLAGS.remove(flag)
    except ValueError:
      pass

ROOT_DIR = os.path.dirname(__file__)

remove_unwanted_pytorch_nvcc_flags()
ext_modules = []
ext_modules.append(
    torch_cpp_ext.CUDAExtension(
        "atmm_ops",
        sources=[
            "valora/csrc/lora_ops.cc",
        ] + glob("valora/csrc/bgmv/*.cu"),
        include_dirs=[
            ROOT_DIR+"/valora/csrc/bgmv/third_party/cutlass/include/",
        ],
        extra_compile_args={
            'cxx': ['-std=c++17'],
            'nvcc': [
                '-std=c++17',
            ]
        },
    ))

setup(
    name="atmm_ops",
    ext_modules=ext_modules,
    cmdclass={"build_ext": torch_cpp_ext.BuildExtension},
)
