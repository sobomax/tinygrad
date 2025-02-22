from __future__ import annotations
from importlib.util import find_spec as f_s, module_from_spec as m_f_s
import os, functools, platform, time, re, contextlib, operator, hashlib, pickle, sqlite3, cProfile, pstats, tempfile, pathlib, string, ctypes
from urllib import request  # NOTE: this has to be imported specifically
from tqdm import tqdm
from typing import Dict, Tuple, Union, List, ClassVar, Optional, Iterable, Any, TypeVar, TYPE_CHECKING, Callable, Sequence
if TYPE_CHECKING:  # TODO: remove this and import TypeGuard from typing once minimum python supported version is 3.10
  from typing_extensions import TypeGuard

def try_import(mod, sym): return s if (ms:=f_s(mod)) and (m:=m_f_s(ms)) and not ms.loader.exec_module(m) and (s:=getattr(m, sym, None)) else None

T = TypeVar("T")
U = TypeVar("U")
# NOTE: it returns int 1 if x is empty regardless of the type of x
def prod(x:Iterable[T]) -> Union[T,int]: return functools.reduce(operator.mul, x, 1)

# NOTE: helpers is not allowed to import from anything else in tinygrad
OSX = platform.system() == "Darwin"
CI = os.getenv("CI", "") != ""

def dedup(x:Iterable[T]): return list(dict.fromkeys(x))   # retains list order
def argfix(*x): return tuple(x[0]) if x and x[0].__class__ in (tuple, list) else x
def argsort(x): return type(x)(sorted(range(len(x)), key=x.__getitem__)) # https://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python
def all_same(items:List[T]): return all(x == items[0] for x in items)
def all_int(t: Sequence[Any]) -> TypeGuard[Tuple[int, ...]]: return all(isinstance(s, int) for s in t)
def colored(st, color:Optional[str], background=False): return f"\u001b[{10*background+60*(color.upper() == color)+30+['black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'].index(color.lower())}m{st}\u001b[0m" if color is not None else st  # replace the termcolor library with one line  # noqa: E501
def ansistrip(s:str): return re.sub('\x1b\\[(K|.*?m)', '', s)
def ansilen(s:str): return len(ansistrip(s))
def make_pair(x:Union[int, Tuple[int, ...]], cnt=2) -> Tuple[int, ...]: return (x,)*cnt if isinstance(x, int) else x
def flatten(l:Iterable[Iterable[T]]): return [item for sublist in l for item in sublist]
def fully_flatten(l): return [item for sublist in l for item in (fully_flatten(sublist) if isinstance(sublist, (tuple, list)) else [sublist])]
def fromimport(mod, frm): return getattr(__import__(mod, fromlist=[frm]), frm)
def strip_parens(fst:str): return fst[1:-1] if fst[0] == '(' and fst[-1] == ')' and fst[1:-1].find('(') <= fst[1:-1].find(')') else fst
def round_up(num, amt:int): return (num+amt-1)//amt * amt
def merge_dicts(ds:Iterable[Dict[T,U]]) -> Dict[T,U]:
  assert len(kvs:=set([(k,v) for d in ds for k,v in d.items()])) == len(set(kv[0] for kv in kvs)), f"cannot merge, {kvs} contains different values for the same key"  # noqa: E501
  return {k:v for d in ds for k,v in d.items()}
def partition(lst:List[T], fxn:Callable[[T],bool]) -> Tuple[List[T], List[T]]:
  a:List[T] = []
  b:List[T] = []
  for s in lst: (a if fxn(s) else b).append(s)
  return a,b
def unwrap(x:Optional[T]) -> T:
  assert x is not None
  return x
def unwrap2(x:Tuple[T,Any]) -> T:
  ret, err = x
  assert err is None, str(err)
  return ret
def get_child(obj, key):
  for k in key.split('.'):
    if k.isnumeric(): obj = obj[int(k)]
    elif isinstance(obj, dict): obj = obj[k]
    else: obj = getattr(obj, k)
  return obj

@functools.lru_cache(maxsize=None)
def to_function_name(s:str): return ''.join([c if c in (string.ascii_letters+string.digits+'_') else f'{ord(c):02X}' for c in ansistrip(s)])
@functools.lru_cache(maxsize=None)
def getenv(key:str, default=0): return type(default)(os.getenv(key, default))
def temp(x:str) -> str: return (pathlib.Path(tempfile.gettempdir()) / x).as_posix()

class Context(contextlib.ContextDecorator):
  stack: ClassVar[List[dict[str, int]]] = [{}]
  def __init__(self, **kwargs): self.kwargs = kwargs
  def __enter__(self):
    Context.stack[-1] = {k:o.value for k,o in ContextVar._cache.items()} # Store current state.
    for k,v in self.kwargs.items(): ContextVar._cache[k].value = v # Update to new temporary state.
    Context.stack.append(self.kwargs) # Store the temporary state so we know what to undo later.
  def __exit__(self, *args):
    for k in Context.stack.pop(): ContextVar._cache[k].value = Context.stack[-1].get(k, ContextVar._cache[k].value)

class ContextVar:
  _cache: ClassVar[Dict[str, ContextVar]] = {}
  value: int
  def __new__(cls, key, default_value):
    if key in ContextVar._cache: return ContextVar._cache[key]
    instance = ContextVar._cache[key] = super().__new__(cls)
    instance.value = getenv(key, default_value)
    return instance
  def __bool__(self): return bool(self.value)
  def __ge__(self, x): return self.value >= x
  def __gt__(self, x): return self.value > x
  def __lt__(self, x): return self.value < x

DEBUG, IMAGE, BEAM, NOOPT = ContextVar("DEBUG", 0), ContextVar("IMAGE", 0), ContextVar("BEAM", 0), ContextVar("NOOPT", 0)
GRAPH, GRAPHPATH = ContextVar("GRAPH", 0), getenv("GRAPHPATH", "/tmp/net")

class Timing(contextlib.ContextDecorator):
  def __init__(self, prefix="", on_exit=None, enabled=True): self.prefix, self.on_exit, self.enabled = prefix, on_exit, enabled
  def __enter__(self): self.st = time.perf_counter_ns()
  def __exit__(self, *exc):
    self.et = time.perf_counter_ns() - self.st
    if self.enabled: print(f"{self.prefix}{self.et*1e-6:6.2f} ms"+(self.on_exit(self.et) if self.on_exit else ""))

def _format_fcn(fcn): return f"{fcn[0]}:{fcn[2]}" if fcn[2] != "<genexpr>" else f"{fcn[0]}:{fcn[1]}"
class Profiling_default(contextlib.ContextDecorator):
  def __init__(self, enabled=True, sort='cumtime', frac=0.2, fn=None, ts=1):
    self.enabled, self.sort, self.frac, self.fn, self.time_scale = enabled, sort, frac, fn, 1e3/ts
  def __enter__(self):
    self.pr = cProfile.Profile()
    if self.enabled: self.pr.enable()
  def __exit__(self, *exc):
    if self.enabled:
      self.pr.disable()
      if self.fn: self.pr.dump_stats(self.fn)
      stats = pstats.Stats(self.pr).strip_dirs().sort_stats(self.sort)
      for fcn in stats.fcn_list[0:int(len(stats.fcn_list)*self.frac)]:    # type: ignore[attr-defined]
        (_primitive_calls, num_calls, tottime, cumtime, callers) = stats.stats[fcn]    # type: ignore[attr-defined]
        scallers = sorted(callers.items(), key=lambda x: -x[1][2])
        print(f"n:{num_calls:8d}  tm:{tottime*self.time_scale:7.2f}ms  tot:{cumtime*self.time_scale:7.2f}ms",
              colored(_format_fcn(fcn), "yellow") + " "*(50-len(_format_fcn(fcn))),
              colored(f"<- {(scallers[0][1][2]/tottime)*100:3.0f}% {_format_fcn(scallers[0][0])}", "BLACK") if len(scallers) else '')

class Profiling_pyinst(Profiling_default):
  def __enter__(self): self.pr, _ = ((pr:=Profiler()), pr.start()) if self.enabled else (None, None)
  def __exit__(self, *exc): _ = self.pr.stop() and print(self.pr.output_text(unicode=True, color=True))

Profiling = Profiling_pyinst if os.getenv("PYINST", False) and (Profiler:=try_import('pyinstrument', 'Profiler')) else Profiling_default

# *** universal database cache ***

_cache_dir: str = getenv("XDG_CACHE_HOME", os.path.expanduser("~/Library/Caches" if OSX else "~/.cache"))
CACHEDB: str = getenv("CACHEDB", os.path.abspath(os.path.join(_cache_dir, "tinygrad", "cache.db")))
CACHELEVEL = getenv("CACHELEVEL", 2)

VERSION = 10
_db_connection = None
def db_connection():
  global _db_connection
  if _db_connection is None:
    os.makedirs(CACHEDB.rsplit(os.sep, 1)[0], exist_ok=True)
    _db_connection = sqlite3.connect(CACHEDB)
    if DEBUG >= 7: _db_connection.set_trace_callback(print)
  return _db_connection

def diskcache_get(table:str, key:Union[Dict, str, int]) -> Any:
  if CACHELEVEL == 0: return None
  if isinstance(key, (str,int)): key = {"key": key}
  conn = db_connection()
  cur = conn.cursor()
  try:
    res = cur.execute(f"SELECT val FROM {table}_{VERSION} WHERE {' AND '.join([f'{x}=?' for x in key.keys()])}", tuple(key.values()))
  except sqlite3.OperationalError:
    return None  # table doesn't exist
  if (val:=res.fetchone()) is not None: return pickle.loads(val[0])
  return None

_db_tables = set()
def diskcache_put(table:str, key:Union[Dict, str, int], val:Any):
  if CACHELEVEL == 0: return val
  if isinstance(key, (str,int)): key = {"key": key}
  conn = db_connection()
  cur = conn.cursor()
  if table not in _db_tables:
    TYPES = {str: "text", bool: "integer", int: "integer", float: "numeric", bytes: "blob"}
    ltypes = ', '.join(f"{k} {TYPES[type(key[k])]}" for k in key.keys())
    cur.execute(f"CREATE TABLE IF NOT EXISTS {table}_{VERSION} ({ltypes}, val blob, PRIMARY KEY ({', '.join(key.keys())}))")
    _db_tables.add(table)
  cur.execute(f"REPLACE INTO {table}_{VERSION} ({', '.join(key.keys())}, val) VALUES ({', '.join(['?']*len(key.keys()))}, ?)", tuple(key.values()) + (pickle.dumps(val), ))  # noqa: E501
  conn.commit()
  cur.close()
  return val

# *** http support ***

def fetch(url:str, name:Optional[Union[pathlib.Path, str]]=None, allow_caching=not getenv("DISABLE_HTTP_CACHE")) -> pathlib.Path:
  if url.startswith(("/", ".")): return pathlib.Path(url)
  fp = pathlib.Path(name) if name is not None and (isinstance(name, pathlib.Path) or '/' in name) else pathlib.Path(_cache_dir) / "tinygrad" / "downloads" / (name if name else hashlib.md5(url.encode('utf-8')).hexdigest())  # noqa: E501
  if not fp.is_file() or not allow_caching:
    with request.urlopen(url, timeout=10) as r:
      assert r.status == 200
      total_length = int(r.headers.get('content-length', 0))
      progress_bar = tqdm(total=total_length, unit='B', unit_scale=True, desc=url)
      (path := fp.parent).mkdir(parents=True, exist_ok=True)
      with tempfile.NamedTemporaryFile(dir=path, delete=False) as f:
        while chunk := r.read(16384): progress_bar.update(f.write(chunk))
        f.close()
        if (file_size:=os.stat(f.name).st_size) < total_length: raise RuntimeError(f"fetch size incomplete, {file_size} < {total_length}")
        pathlib.Path(f.name).rename(fp)
  return fp

# *** Exec helpers

def cpu_time_execution(cb, enable):
  if enable: st = time.perf_counter()
  cb()
  if enable: return time.perf_counter()-st

# *** ctypes helpers

# TODO: make this work with read only memoryviews (if possible)
def from_mv(mv:memoryview, to_type=ctypes.c_char): return ctypes.cast(ctypes.addressof(to_type.from_buffer(mv)), ctypes.POINTER(to_type))
def to_mv(ptr, sz) -> memoryview: return memoryview(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint8 * sz)).contents).cast("B")
def to_char_p_p(options: List[bytes], to_type=ctypes.c_char): return (ctypes.POINTER(to_type) * len(options))(*[ctypes.cast(ctypes.create_string_buffer(o), ctypes.POINTER(to_type)) for o in options])  # noqa: E501
@functools.lru_cache(maxsize=None)
def init_c_struct_t(fields: Tuple[Tuple[str, ctypes._SimpleCData], ...]):
  class CStruct(ctypes.Structure):
    _pack_, _fields_ = 1, fields
  return CStruct
def init_c_var(ctypes_var, creat_cb): return (creat_cb(ctypes_var), ctypes_var)[1]
def get_bytes(arg, get_sz, get_str, check) -> bytes: return (sz := init_c_var(ctypes.c_size_t(), lambda x: check(get_sz(arg, ctypes.byref(x)))), ctypes.string_at(init_c_var(ctypes.create_string_buffer(sz.value), lambda x: check(get_str(arg, x))), size=sz.value))[1]  # noqa: E501
def flat_mv(mv:memoryview):
  if len(mv) == 0: return mv
  return mv.cast("B", shape=(mv.nbytes,))

# *** Helpers for CUDA-like APIs.

def compile_cuda_style(prg, compile_options, prog_t, create_prog, compile_prog, get_code, get_code_size, get_log, get_log_size, check) -> bytes:
  check(create_prog(ctypes.byref(prog := prog_t()), prg.encode(), "<null>".encode(), 0, None, None))
  status = compile_prog(prog, len(compile_options), to_char_p_p([o.encode() for o in compile_options]))

  if status != 0: raise RuntimeError(f"compile failed: {get_bytes(prog, get_log_size, get_log, check).decode()}")
  return get_bytes(prog, get_code_size, get_code, check)

def encode_args_cuda_style(bufs, vals, device_ptr_t, marks) -> Tuple[ctypes.Array, ctypes.Structure]:
  c_args = init_c_struct_t(tuple([(f'f{i}', device_ptr_t) for i in range(len(bufs))] + [(f'f{i}', ctypes.c_int) for i in range(len(bufs), len(bufs)+len(vals))]))(*bufs, *vals)  # noqa: E501
  return (ctypes.c_void_p * 5)(ctypes.c_void_p(marks[0]), ctypes.cast(ctypes.pointer(c_args), ctypes.c_void_p), ctypes.c_void_p(marks[1]), ctypes.cast(ctypes.pointer(ctypes.c_size_t(ctypes.sizeof(c_args))), ctypes.c_void_p), ctypes.c_void_p(marks[2])), c_args  # noqa: E501

def time_execution_cuda_style(cb, ev_t, evcreate, evrecord, evsync, evdestroy, evtime, enable=False) -> Optional[float]:
  if not enable: return cb()
  evs = [init_c_var(ev_t(), lambda x: evcreate(ctypes.byref(x), 0)) for _ in range(2)]
  evrecord(evs[0], None)
  cb()
  evrecord(evs[1], None)
  evsync(evs[1])
  evtime(ctypes.byref(ret := ctypes.c_float()), evs[0], evs[1])
  for ev in evs: evdestroy(ev)
  return ret.value * 1e-3
