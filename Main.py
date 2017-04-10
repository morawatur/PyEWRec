from numba import cuda
import Constants as const
import CudaConfig as ccfg
import GUI as gui

cuda.select_device(0)
dev = cuda.get_current_device()
print('CUDA device in use: ' + dev.name.decode())

ccfg.GetGPUFreeMemory()
ccfg.GetGPUMemoryUsed()

gui.RunEwrWindow(const.gridDim)