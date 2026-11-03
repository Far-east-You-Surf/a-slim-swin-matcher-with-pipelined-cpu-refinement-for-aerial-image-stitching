"""Runner TensorRT tanpa torch (cuda-python `cuda.bindings`), supaya skrip yang SAMA
berjalan di mesin lokal dan di kontainer NGC TensorRT (pip install cuda-python)."""
import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cr


def ck(ret):
    err = ret[0] if isinstance(ret, tuple) else ret
    if int(err) != 0:
        raise RuntimeError(f"CUDA error {err}")
    return ret[1:] if isinstance(ret, tuple) and len(ret) > 1 else None


class Runner:
    def __init__(self, engine_path, seed=0):
        self.rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open(engine_path, "rb") as f:
            self.engine = self.rt.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.stream = ck(cr.cudaStreamCreate())[0]
        rng = np.random.default_rng(seed)
        self.ptrs = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shp = tuple(self.engine.get_tensor_shape(name))
            dt = trt.nptype(self.engine.get_tensor_dtype(name))
            nbytes = int(np.prod(shp)) * np.dtype(dt).itemsize
            ptr = ck(cr.cudaMalloc(nbytes))[0]
            self.ptrs.append(ptr)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                host = rng.standard_normal(shp).astype(dt)  # masukan acak
                ck(cr.cudaMemcpy(ptr, host.ctypes.data, nbytes, cr.cudaMemcpyKind.cudaMemcpyHostToDevice))
            self.ctx.set_tensor_address(name, int(ptr))

    def launch(self, n=1):
        for _ in range(n):
            self.ctx.execute_async_v3(int(self.stream))

    def sync(self):
        ck(cr.cudaStreamSynchronize(self.stream))

    def close(self):
        for p in self.ptrs:
            cr.cudaFree(p)
        cr.cudaStreamDestroy(self.stream)
