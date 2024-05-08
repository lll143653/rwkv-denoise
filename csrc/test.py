T_MAX = 65536
HEAD_SIZE = 64
import os
from torch.utils.cpp_extension import load
wkv6_cuda = load(name="wkv6", sources=[f"{os.path.dirname(__file__)}/cuda/wkv_op.cpp", f"{os.path.dirname(__file__)}/cuda/wkv_cuda.cu"],
                 verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization",f"-D_N_={HEAD_SIZE}", 
                 f"-D_T_={T_MAX}"])