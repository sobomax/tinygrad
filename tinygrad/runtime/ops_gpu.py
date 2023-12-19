from __future__ import annotations
from typing import Tuple, Optional, List
import ctypes, functools
import gpuctypes.opencl as cl
from tinygrad.helpers import init_c_var, to_char_p_p, from_mv, OSX, DEBUG
from tinygrad.dtype import ImageDType
from tinygrad.codegen.kernel import LinearizerOptions
from tinygrad.renderer.cstyle import OpenCLRenderer
from tinygrad.device import Compiled, LRUAllocator

# see test/external/external_osx_profiling.py to determine this ratio. it's in like GPU clocks or something
OSX_TIMING_RATIO = (125/3) if OSX else 1.0

def check(status):
  if status != 0: raise RuntimeError(f"OpenCL Error {status}")
def checked(ret, status): return (check(status.value), ret)[1]

def cl_auto(ap, *a, _i=init_c_var, _c=ctypes.c_uint32, _ot=ctypes.c_uint32, c1=check, c2=check):
  return (_i((_ot * n.value)(), lambda x: c2(ap(*a, n.value, x, None))), n) if (n:=_i(_c(), lambda x: c1(ap(*a, 0, None, ctypes.byref(x))))) else None

def compile_cl(prg:str) -> bytes:
  assert CLDevice.c_ctx is not None, 'OpenCL requires a "c_ctx" to compile, init a device before you call this'
  program = checked(cl.clCreateProgramWithSource(CLDevice.c_ctx.context, 1, to_char_p_p([prg_bytes := prg.encode()]),
                                                 ctypes.byref(ctypes.c_size_t(len(prg_bytes))), ctypes.byref(status := ctypes.c_int32())), status)
  status = cl.clBuildProgram(program, 1, ctypes.byref(CLDevice.c_ctx.device_id), None, cl.clBuildProgram.argtypes[4](), None)
  if status != 0:
    ms, ls = cl_auto(cl.clGetProgramBuildInfo, program, CLDevice.c_ctx.device_id, cl.CL_PROGRAM_BUILD_LOG, _ot=ctypes.c_char, _c=ctypes.c_size_t)
    raise RuntimeError(f"OpenCL Compile Error\n\n{ctypes.string_at(ms, size=ls.value).decode()}")
  binary_sizes = init_c_var((ctypes.c_size_t * 1)(), lambda x: check(cl.clGetProgramInfo(program, cl.CL_PROGRAM_BINARY_SIZES, ctypes.sizeof(x), ctypes.byref(x), None)))  # noqa: E501
  binary = init_c_var(ctypes.create_string_buffer(binary_sizes[0]), lambda x: check(cl.clGetProgramInfo(program, cl.CL_PROGRAM_BINARIES, ctypes.sizeof(ctypes.c_void_p), ctypes.byref((ctypes.c_void_p * 1)(ctypes.addressof(x))), None)))  # noqa: E501
  check(cl.clReleaseProgram(program))
  return bytes(binary)

class CLProgram:
  def __init__(self, device:CLDevice, name:str, lib:bytes):
    self.device, self.name, self.lib = device, name, lib
    self.program = checked(cl.clCreateProgramWithBinary(device.context, 1, ctypes.byref(device.device_id), (ctypes.c_size_t * 1)(len(lib)),
                                                        to_char_p_p([lib], ctypes.c_ubyte), ctypes.byref(binary_status := ctypes.c_int32()),
                                                        ctypes.byref(errcode_ret := ctypes.c_int32())), errcode_ret)
    check(binary_status.value)
    check(cl.clBuildProgram(self.program, 1, ctypes.byref(device.device_id), None, cl.clBuildProgram.argtypes[4](), None)) # NOTE: OSX requires this
    self.kernel = checked(cl.clCreateKernel(self.program, name.encode(), ctypes.byref(status := ctypes.c_int32())), status)

  def __del__(self):
    if hasattr(self, 'kernel'): check(cl.clReleaseKernel(self.kernel))
    if hasattr(self, 'program'): check(cl.clReleaseProgram(self.program))

  def __call__(self, *bufs:cl.cl_mem, global_size:Tuple[int,...], local_size:Optional[Tuple[int,...]]=None, vals:Tuple[int, ...]=(), wait=False) -> Optional[float]:  # noqa: E501
    for i,b in enumerate(bufs): cl.clSetKernelArg(self.kernel, i, ctypes.sizeof(b), ctypes.byref(b))
    for i,b in enumerate(vals,start=len(bufs)): cl.clSetKernelArg(self.kernel, i, 4, ctypes.byref(ctypes.c_int32(b)))
    if local_size is not None: global_size = tuple(int(g*l) for g,l in zip(global_size, local_size))
    event = cl.cl_event() if wait else None
    check(cl.clEnqueueNDRangeKernel(self.device.queue, self.kernel, len(global_size), None, (ctypes.c_size_t * len(global_size))(*global_size), (ctypes.c_size_t * len(local_size))(*local_size) if local_size else None, 0, None, event))  # noqa: E501
    if not wait: return
    check(cl.clWaitForEvents(1, ctypes.byref(event)))
    start = init_c_var(ctypes.c_uint64(), lambda x: check(cl.clGetEventProfilingInfo(event, cl.CL_PROFILING_COMMAND_START, ctypes.sizeof(x), ctypes.byref(x), None)))  # noqa: E501
    end = init_c_var(ctypes.c_uint64(), lambda x: check(cl.clGetEventProfilingInfo(event, cl.CL_PROFILING_COMMAND_END, ctypes.sizeof(x), ctypes.byref(x), None)))  # noqa: E501
    return float(end.value-start.value) * OSX_TIMING_RATIO * 1e-9

class CLAllocator(LRUAllocator):
  def __init__(self, device:CLDevice):
    self.device = device
    super().__init__()
  def _alloc(self, size:int) -> cl.cl_mem:
    return checked(cl.clCreateBuffer(self.device.context, cl.CL_MEM_READ_WRITE, size, None, ctypes.byref(status := ctypes.c_int32())), status)
  def _alloc_image(self, dtype:ImageDType) -> cl.cl_mem:
    return checked(cl.clCreateImage2D(self.device.context, cl.CL_MEM_READ_WRITE,
                                      cl.cl_image_format(cl.CL_RGBA, {2: cl.CL_HALF_FLOAT, 4: cl.CL_FLOAT}[dtype.itemsize]),
                                      dtype.shape[1], dtype.shape[0], 0, None, ctypes.byref(status := ctypes.c_int32())), status)
  def _free(self, buf:cl.cl_mem): check(cl.clReleaseMemObject(buf))
  def copyin(self, dest:cl.cl_mem, src:memoryview):
    check(cl.clEnqueueWriteBuffer(self.device.queue, dest, False, 0, len(src)*src.itemsize, from_mv(src), 0, None, None))
    self.device.pending_copyin.append(src)    # NOTE: these can't be freed until the GPU actually executes this command
  def copyout(self, dest:memoryview, src:cl.cl_mem):
    check(cl.clEnqueueReadBuffer(self.device.queue, src, False, 0, len(dest)*dest.itemsize, from_mv(dest), 0, None, None))
    self.device.synchronize()

class CLDevice(Compiled):
  # device_ids is global and only initted once. c_ctx is the first created context, we assume they are all the same for the compiler
  device_ids = c_ctx = None
  def __init__(self, device:str=""):
    chosen_device_id = 1 if ":" not in device else int(device.split(":")[1])
    if CLDevice.device_ids is None:
      platform_ids, num_platforms = cl_auto(cl.clGetPlatformIDs, _ot=cl.cl_platform_id)
      usable_devices = []
      for i, pid in enumerate(platform_ids):
        for dtype in [cl.CL_DEVICE_TYPE_GPU, cl.CL_DEVICE_TYPE_DEFAULT]:
          devices, ndevices = r if (r:=cl_auto(cl.clGetDeviceIDs, pid, dtype, _ot=cl.cl_device_id, c1=lambda x: x)) else (None, None)
          if not ndevices or ndevices.value == 0: continue
          usable_devices.extend([devices[i] for i in range(int(ndevices.value))])
          break
      if DEBUG >= 1: print(f"CLDevice: got {num_platforms.value} platforms and {len(usable_devices)} devices")
      for i, device_id in enumerate(usable_devices if DEBUG >= 1 else ()):
        name_buffer, _ = cl_auto(cl.clGetDeviceInfo, device_id, cl.CL_DEVICE_NAME, _ot=ctypes.c_char, _c=ctypes.c_size_t)
        print(f"Device {'*' if (chosen_device_id - 1) == i else ' '}{i+1}: {name_buffer.value.decode()}")
      CLDevice.device_ids = tuple(usable_devices)
    d = self.device_id = CLDevice.device_ids[chosen_device_id-1]
    self.context = checked((f:=cl.clCreateContext)(None, 1, ctypes.byref(d), f.argtypes[3](), None, ctypes.byref(s:=ctypes.c_int32())), s)  # noqa: E501
    if CLDevice.c_ctx is None: CLDevice.c_ctx = self
    self.queue = checked(cl.clCreateCommandQueue(self.context, d, cl.CL_QUEUE_PROFILING_ENABLE, ctypes.byref(s)), s)
    self.pending_copyin: List[memoryview] = []
    super().__init__(CLAllocator(self), LinearizerOptions("GPU"), OpenCLRenderer, compile_cl, functools.partial(CLProgram, self))
  def synchronize(self):
    check(cl.clFinish(self.queue))
    self.pending_copyin.clear()

GPUDevice = CLDevice # for legacy reasons
