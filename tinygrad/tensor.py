# inspired by https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
from inspect import signature
import numpy as np
import os
try:
  import pyopencl as cl
  import pyopencl.array as cl_array
  GPU = True
except ImportError:
  # no GPU support
  GPU = False

# **** profiler, 10 lines too long ****
DEBUG = os.getenv("DEBUG", None) is not None
if DEBUG:
  import collections, atexit, time
  debug_counts = collections.defaultdict(int)
  debug_times = collections.defaultdict(float)
  def print_debug_exit():
    for name, _ in sorted(debug_times.items(), key=lambda x: -x[1]):
      print("%20s : %3d  %10.2f ms" % (name, debug_counts[name], debug_times[name]))
  atexit.register(print_debug_exit)

class ProfileOp:
  def __init__(self, name, x, backward=False):
    self.name = ("back_" if backward else "")+name
    self.x = x
  def __enter__(self):
    if DEBUG: self.st = time.time()
  def __exit__(self, *junk):
    if DEBUG:
      et = (time.time()-self.st)*1000.
      debug_counts[self.name] += 1
      debug_times[self.name] += et
      print("%20s : %7.2f ms  %s" % (self.name, et, [y.shape for y in self.x]))

cl_ctx, cl_queue = None, None
def require_init_gpu():
  global cl_ctx, cl_queue
  if cl_queue is None:
    try:
      # for Macbook 16 inch
      cl_ctx = cl.create_some_context(answers=[0,2])
    except (cl._cl.RuntimeError, cl._cl.LogicError, TypeError):
      cl_ctx = cl.create_some_context(interactive=False)
    cl_queue = cl.CommandQueue(cl_ctx)

# **** start with two base classes ****

class Tensor:
  did_float_warning = False

  def __init__(self, data, gpu=False):
    if isinstance(data, list):
      data = np.array(data, dtype=np.float32)
    elif GPU and isinstance(data, cl_array.Array):
      self.gpu = True
    elif not isinstance(data, np.ndarray):
      raise TypeError("Error constructing tensor with %r" % data)

    if isinstance(data, np.ndarray):
      if data.dtype != np.float32 and not Tensor.did_float_warning:
        # warning? float64 is actually needed for numerical jacobian
        print("warning, %r isn't float32" % (data.shape,))
        Tensor.did_float_warning = True
      self.gpu = False

    self.data = data
    self.grad = None

    if gpu:
      self.cuda_()

    # internal variables used for autograd graph construction
    self._ctx = None

  def __repr__(self):
    return "Tensor %r with grad %r" % (self.data, self.grad.data if self.grad else None)

  def assign(self, x):
    self.data = x.data

  @property
  def shape(self):
    return self.data.shape

  @classmethod
  def zeros(cls, *shape, gpu=False):
    return cls(np.zeros(shape, dtype=np.float32), gpu=gpu)

  @classmethod
  def ones(cls, *shape, gpu=False):
    return cls(np.ones(shape, dtype=np.float32), gpu=gpu)

  @classmethod
  def randn(cls, *shape, gpu=False):
    return cls(np.random.randn(*shape).astype(np.float32), gpu=gpu)

  @classmethod
  def eye(cls, dim, gpu=False):
    return cls(np.eye(dim).astype(np.float32), gpu=gpu)

  def backward(self, allow_fill=True):
    if self._ctx is None:
      return

    if self.grad is None and allow_fill:
      # fill in the first grad with one
      # this is "implicit gradient creation"
      assert self.data.shape == (1,)
      self.grad = Tensor.ones(*self.data.shape, gpu=self.gpu)

    visited, nodes = set(), []
    def deepwalk(node):
      visited.add(node)
      if node._ctx:
        for i in node._ctx.parents:
          if i not in visited:
            deepwalk(i)
        nodes.append(node)
    deepwalk(self)

    for t0 in reversed(nodes):
      assert (t0.grad is not None)
      with ProfileOp(t0._ctx.__class__.__name__, [t0.grad], backward=True):
        grads = t0._ctx.backward(t0._ctx, t0.grad.data)
      if len(t0._ctx.parents) == 1:
        grads = [grads]
      for t,g in zip(t0._ctx.parents, grads):
        if g is None:
          continue
        assert g.shape == t.data.shape, \
          "grad shape must match tensor shape in %r, %r != %r" % (self._ctx, g.shape, t.data.shape)
        t.grad = Tensor(g) if t.grad is None else (t.grad + Tensor(g))

  # ***** tinygrad supports CPU and GPU *****

  def cpu(self):
    if self.gpu:
      ret = Tensor(self.data.get(cl_queue), gpu=False)
      if self.grad:
        ret.grad = self.grad.cpu()
      return ret
    return self

  def cuda_(self):
    self.data = self.cuda().data
    self.gpu = True

  def cuda(self):
    if not GPU:
      raise Exception("No GPU Support, install pyopencl")
    if not self.gpu:
      require_init_gpu()
      assert self.data.dtype == np.float32   # only float32 on GPU
      ret = Tensor(cl_array.to_device(cl_queue, self.data), gpu=True)
      if self.grad:
        ret.grad = self.grad.cuda()
      return ret
    return self

  def detach(self):
    return Tensor(self.data, self.gpu)

  # ***** put ops in these dicts *****

  ops = {}
  opsgpu = {}

  # ***** non first class ops *****

  def mean(self):
    div = Tensor(np.array([1/np.prod(self.shape)], dtype=self.data.dtype), gpu=self.gpu)
    return self.sum().mul(div)

  def sqrt(self):
    root = Tensor(np.zeros(self.shape, dtype=self.data.dtype)+0.5, gpu=self.gpu)
    return self.pow(root)

  def div(self, y):
    root = Tensor(np.zeros(self.shape, dtype=self.data.dtype)-1, gpu=self.gpu)
    return self.mul(y.pow(root))

  def swish(self):
    return self.mul(self.sigmoid())

  def tanh(self):
    t2 = Tensor(np.zeros(self.shape, dtype=self.data.dtype)+2, gpu=self.gpu)
    t1 = Tensor(np.zeros(self.shape, dtype=self.data.dtype)+1, gpu=self.gpu)
    return self.mul(t2).sigmoid().mul(t2) - t1 # 2*sigmoid(2*x)-1

# An instantiation of the Function is the Context
class Function:
  def __init__(self, *tensors):
    self.parents = tensors
    self.saved_tensors = []

  def save_for_backward(self, *x):
    self.saved_tensors.extend(x)

  def apply(self, *x, **kwargs):
    op = self
    ctx = op(*x)
    # use default params
    params = signature(op.forward).parameters
    for p in params.values():
      if p.default is not p.empty:
        setattr(ctx, p.name, p.default)
    # overwrite with passed params
    for k, v in kwargs.items():
      setattr(ctx, k, v)
    with ProfileOp(ctx.__class__.__name__, x):
      ret = Tensor(op.forward(ctx, *[t.data for t in x], **kwargs))
    ret._ctx = ctx
    return ret

def register(name, fxn, gpu=False):
  if gpu:
    Tensor.opsgpu[name] = fxn
  else:
    Tensor.ops[name] = fxn
  def dispatch(*x, **kwargs):
    f = (Tensor.opsgpu if x[0].gpu else Tensor.ops)[name]
    f.cl_ctx, f.cl_queue = cl_ctx, cl_queue
    return f.apply(f, *x, **kwargs)
  setattr(Tensor, name, dispatch)
  if name in ['add', 'sub', 'mul', 'div']:
    setattr(Tensor, "__%s__" % name, dispatch)
    setattr(Tensor, "__i%s__" % name, lambda self,x: self.assign(dispatch(self,x)))


# this registers all the operations
import tinygrad.ops
if GPU:
  import tinygrad.opsgpu

