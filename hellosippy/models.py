import tinygrad.nn as nn
from tinygrad.tensor import Tensor
from tinygrad.nn.state import torch_load, load_state_dict
from tinygrad.helpers import fetch, Timing
from tinygrad.device import Device

class AmendmentNetwork1Config:
  LRELU_SLOPE = 0.01
  chunk_size = 8
  pre_frames = 2
  post_frames = 2
  frame_size = 256
  num_mels = 80
  chunk_size: int
  trim_pr: int
  trim_po: int
  output_size: int
  eframes: int

  def __init__(self):
    self.eframes = self.pre_frames + self.post_frames
    self.trim_pr = self.pre_frames * self.frame_size
    self.trim_po = self.post_frames * self.frame_size
    self.output_size = self.chunk_size * self.frame_size

class SimpleResidualBlock:
  def __init__(self, channels):
    self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, stride=1,
                            padding=1, dilation=1)
    self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, stride=1,
                            padding=3, dilation=3)

  def __call__(self, x, lrelu):
    assert lrelu is not None
    residual = x
    x = lrelu(x)
    x = self.conv1(x)
    x = lrelu(x)
    x = self.conv2(x)
    x += residual
    return x

class AmendmentNetwork1:
  config_class = AmendmentNetwork1Config
  def __init__(self, config=None):
    if config is None: config = self.config_class()
    _c = self._c = config

    self.conv_pre_m = nn.Conv1d(_c.num_mels, 32, kernel_size=3, stride=1, padding=1)
    self.conv_pre_a = nn.Conv1d(_c.frame_size, 160, kernel_size=3, stride=1, padding=1)
    self.upsampler = (
      nn.ConvTranspose1d(192, 128, kernel_size=8, stride=4, padding=2),
      nn.ConvTranspose1d(128, 64, kernel_size=8, stride=4, padding=2),
    )
    self.resblock = SimpleResidualBlock(64)
    self.post_conv = nn.Conv1d(in_channels=64, out_channels=_c.frame_size,
                                kernel_size=8, stride=24, padding=0)

  def lrelu(self, x: Tensor):
    return x.leakyrelu(self._c.LRELU_SLOPE)

  def forward(self, mel: Tensor, audio: Tensor):
    batch_size = audio.shape[0]
    T = mel.shape[-1]
    #print(Exception(f"BP: ms:{mel.size()} as:{audio.size()}"))
    audio_reshaped = audio.reshape(batch_size, self._c.frame_size, -1)
    mel = mel.reshape(batch_size, T, -1)
    #print(Exception(f"BP: ms:{mel.size()} as:{audio.size()} ars:{audio_reshaped.size()}"))
    x_mel = self.conv_pre_m(mel)
    x_audio = self.conv_pre_a(audio_reshaped)
    am_comb = x_mel.cat(x_audio, dim=1)
    for layer in self.upsampler:
      am_comb = self.lrelu(am_comb)
      am_comb = layer(am_comb)
    am_comb = self.resblock(am_comb, self.lrelu)
    am_comb = self.lrelu(am_comb)
    am_comb = self.post_conv(am_comb).squeeze(-1)
    am_comb = self.lrelu(am_comb).reshape(batch_size, -1)
    audio = audio[:, self._c.trim_pr:-self._c.trim_po] * am_comb
    return audio.tanh()

def get_model():
  _m = AmendmentNetwork1()
  d = torch_load(fetch('https://huggingface.co/sobomax/speecht5-rt.post_vocoder.v2/resolve/main/pytorch_model.bin?download=true'))
  load_state_dict(_m, d, device='CPU')
  return _m
m = get_model()
raise Exception("BP")
from time import sleep
from sys import stderr
sleep(1)
batch_size = 1
test_data = []
print("Generating test data, sit tight...")
with Timing("Test Data Generation: "):
  while batch_size <= 8 * 1024:
    msg = f"{batch_size=}\r"
    stderr.write(f"{batch_size=}\r")
    inif = Tensor.randn
    #inif = Tensor.zeros
    #print(f'{[[(batch_size, *_d, "device = CPU") for _d in ((12, 80), (256 * 12))] for _ in range(5)]=}')
    #raise Exception("BP")
    test_data.append((batch_size, [[inif(batch_size, *_d).cpu().realize() for _d in ((12, 80), (256 * 12,))] for _ in range(100)]))
    batch_size *= 2
  else: stderr.write(f"{msg[:-1]}...Done\n")

for batch_size, data in test_data:
  with Timing(f"AmendmentNetwork1({batch_size=}): "):
    for t_mel, t_audio in data:
      m.forward(t_mel.to(Device.DEFAULT), t_audio.to(Device.DEFAULT)).cpu().realize()
      #print(f'{t_mel.shape} {t_audio.shape}')
