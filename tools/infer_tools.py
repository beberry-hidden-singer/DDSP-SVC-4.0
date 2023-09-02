import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ddsp_naive.vocoder import F0Extractor, UnitsEncoder, VolumeExtractor
from diffusion.vocoder import load_model_vocoder, load_model_vocoder_from_combo
from tools.slicer import cross_fade, split
from tools.units_index import UnitsIndexer


class DiffusionSVC:
  def __init__(self, device=None):
    if device is not None:
      self.device = device
    else:
      self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.model_path = None
    self.model = None
    self.vocoder = None
    self.args = None
    # 特征提取器
    self.units_encoder = None
    self.f0_extractor = None
    self.f0_model = None
    self.f0_min = None
    self.f0_max = None
    self.volume_extractor = None
    self.speaker_encoder = None
    self.spk_emb_dict = None
    self.resample_dict_16000 = {}
    self.units_indexer = None
    self.naive_model_path = None
    self.naive_model = None
    self.naive_model_args = None
    self.use_combo_model = False

  def load_model(self, model_path, f0_model=None, f0_min=None, f0_max=None):

    if ('1234' + model_path)[-4:] == '.ptc':
      self.use_combo_model = True
      self.model_path = model_path
      self.naive_model_path = model_path
      diff_model, diff_args, naive_model, naive_args, vocoder = load_model_vocoder_from_combo(model_path, device=self.device)
      self.model = diff_model
      self.args = diff_args
      self.naive_model = naive_model
      self.naive_model_args = naive_args
      self.vocoder = vocoder
    else:
      self.model_path = model_path
      self.model, self.vocoder, self.args = load_model_vocoder(model_path, device=self.device)

    self.units_encoder = UnitsEncoder(
      self.args.data.encoder,
      self.args.data.encoder_ckpt,
      self.args.data.encoder_sample_rate,
      self.args.data.encoder_hop_size,
      cnhubertsoft_gate=self.args.data.cnhubertsoft_gate,
      device=self.device,
    )

    self.volume_extractor = VolumeExtractor(hop_size=512, block_size=self.args.data.block_size)
    self.load_f0_extractor(f0_model=f0_model, f0_min=f0_min, f0_max=f0_max)
    self.units_indexer = UnitsIndexer(os.path.split(model_path)[0])

  def flush(self, model_path=None, f0_model=None, f0_min=None, f0_max=None, naive_model_path=None):
    assert (model_path is not None) or (naive_model_path is not None)
    # flush model if changed
    if ((self.model_path != model_path) or (self.f0_model != f0_model)
            or (self.f0_min != f0_min) or (self.f0_max != f0_max)):
      self.load_model(model_path, f0_model=f0_model, f0_min=f0_min, f0_max=f0_max)
    if (self.naive_model_path != naive_model_path) and (naive_model_path is not None):
      self.load_naive_model(naive_model_path)

    # check args if use naive
    if self.naive_model is not None:
      if self.naive_model_args.data.encoder != self.args.data.encoder:
        raise ValueError("encoder of Naive Model and Diffusion Model are different")
      if self.naive_model_args.model.n_spk != self.args.model.n_spk:
        raise ValueError("n_spk of Naive Model and Diffusion Model are different")
      if bool(self.naive_model_args.model.use_speaker_encoder) != bool(self.args.model.use_speaker_encoder):
        raise ValueError("use_speaker_encoder of Naive Model and Diffusion Model are different")
      if self.naive_model_args.vocoder.type != self.args.vocoder.type:
        raise ValueError("vocoder of Naive Model and Diffusion Model are different")
      if self.naive_model_args.data.block_size != self.args.data.block_size:
        raise ValueError("block_size of Naive Model and Diffusion Model are different")
      if self.naive_model_args.data.sampling_rate != self.args.data.sampling_rate:
        raise ValueError("sampling_rate of Naive Model and Diffusion Model are different")

  def flush_f0_extractor(self, f0_model, f0_min=None, f0_max=None):
    if (f0_model != self.f0_model) and (f0_model is not None):
      self.load_f0_extractor(f0_model)

  def load_f0_extractor(self, f0_model, f0_min=None, f0_max=None):
    self.f0_model = f0_model if (f0_model is not None) else self.args.data.f0_extractor
    self.f0_min = f0_min if (f0_min is not None) else self.args.data.f0_min
    self.f0_max = f0_max if (f0_max is not None) else self.args.data.f0_max
    self.f0_model = f0_model
    self.f0_extractor = F0Extractor(
      f0_extractor=self.f0_model,
      sample_rate=44100,
      hop_size=512,
      f0_min=self.f0_min,
      f0_max=self.f0_max,
    )

  def load_naive_model(self, naive_model_path):
    self.naive_model_path = naive_model_path
    model, _, args = load_model_vocoder(naive_model_path, device=self.device, loaded_vocoder=self.vocoder)
    self.naive_model = model
    self.naive_model_args = args
    print(f" [INFO] Load naive model from {naive_model_path}")

  @torch.no_grad()
  def naive_model_call(self, units, f0, volume, spk_id=1, spk_mix_dict=None, aug_shift=0, spk_emb=None):
    # spk_id
    spk_emb_dict = None
    if self.args.model.use_speaker_encoder:  # with speaker encoder
      spk_mix_dict, spk_emb = self.pre_spk_emb(spk_id, spk_mix_dict, len(units), spk_emb)
    # without speaker encoder
    else:
      spk_id = torch.LongTensor(np.array([[int(spk_id)]])).to(self.device)
    aug_shift = torch.from_numpy(np.array([[float(aug_shift)]])).float().to(self.device)
    out_spec = self.naive_model(units, f0, volume, spk_id=spk_id, spk_mix_dict=spk_mix_dict,
                                aug_shift=aug_shift, infer=True, spk_emb=spk_emb, spk_emb_dict=spk_emb_dict)
    return out_spec

  def set_spk_emb_dict(self, spk_emb_dict_or_path):  # 从路径加载或直接设置
    if spk_emb_dict_or_path is None:
      return None
    if spk_emb_dict_or_path is dict:
      self.spk_emb_dict = spk_emb_dict_or_path
      print(f" [INFO] Load spk_emb_dict from {spk_emb_dict_or_path}")
    else:
      self.spk_emb_dict = np.load(spk_emb_dict_or_path, allow_pickle=True).item()
      print(f" [INFO] Load spk_emb_dict from {spk_emb_dict_or_path}")

  @torch.no_grad()
  def encode_units(self, audio, sr=44100, padding_mask=None):
    assert self.units_encoder is not None
    hop_size = self.args.data.block_size * sr / self.args.data.sampling_rate
    return self.units_encoder.encode(audio, sr, hop_size, padding_mask=padding_mask)

  @torch.no_grad()
  def extract_f0(self, audio, key=0, sr=44100, silence_front=0):
    assert self.f0_extractor is not None
    f0 = self.f0_extractor.extract(audio, uv_interp=True, device=self.device, silence_front=silence_front)
    f0 = torch.from_numpy(f0).float().to(self.device).unsqueeze(-1).unsqueeze(0)
    f0 = f0 * 2 ** (float(key) / 12)
    return f0

  @torch.no_grad()
  def extract_volume_and_mask(self, audio, sr=44100, threhold=-60.0):
    assert self.volume_extractor is not None
    volume = self.volume_extractor.extract(audio)
    mask = self.volume_extractor.get_mask_from_volume(volume, threhold=threhold, device=self.device)
    volume = torch.from_numpy(volume).float().to(self.device).unsqueeze(-1).unsqueeze(0)
    return volume, mask

  @torch.no_grad()
  def extract_mel(self, audio, sr=44100):
    assert sr == 441000
    mel = self.vocoder.extract(audio, self.args.data.sampling_rate)
    return mel

  @torch.no_grad()
  def encode_spk(self, audio, sr=44100):
    assert self.speaker_encoder is not None
    return self.speaker_encoder(audio=audio, sample_rate=sr)

  @torch.no_grad()
  def encode_spk_from_path(self, path):  # 从path读取预先提取的声纹(必须是.npy文件), 或从声音文件提取声纹(此时可以是文件或目录)
    if path is None:
      return None
    assert self.speaker_encoder is not None
    if (('122333444455555' + path)[-4:] == '.npy') and os.path.isfile(path):
      spk_emb = np.load(path)
    else:
      if os.path.isfile(path):
        path_list = [path]
      else:
        path_list = os.listdir(path)
        for _index in range(len(path_list)):
          path_list[_index] = os.path.join(path, path_list[_index])
      spk_emb = self.speaker_encoder.mean_spk_emb_from_path_list(path_list)
    return spk_emb

  def pre_spk_emb(self, spk_id, spk_mix_dict, units_len, spk_emb):
    spk_emb_dict = self.spk_emb_dict
    if (spk_mix_dict is not None) or (spk_emb is None):
      assert spk_emb_dict is not None
    if spk_emb is None:
      spk_emb = spk_emb_dict[str(spk_id)]
    # pad and to device
    spk_emb = np.tile(spk_emb, (units_len, 1))
    spk_emb = torch.from_numpy(spk_emb).float().to(self.device)
    return spk_mix_dict, spk_emb

  @torch.no_grad()
  def mel2wav(self, mel, f0, start_frame=0):
    if start_frame == 0:
      return self.vocoder.infer(mel, f0)
    else:  # for realtime speedup
      mel = mel[:, start_frame:, :]
      f0 = f0[:, start_frame:, :]
      out_wav = self.vocoder.infer(mel, f0)
      return F.pad(out_wav, (start_frame * self.vocoder.vocoder_hop_size, 0))

  @torch.no_grad()  # 最基本推理代码,将输入标准化为tensor,只与mel打交道
  def __call__(self, units, f0, volume, spk_id=1, spk_mix_dict=None, aug_shift=0, gt_spec=None, infer_speedup=10,
               method='dpm-solver', k_step=None, use_tqdm=True, spk_emb=None):

    if self.args.model.k_step_max is not None:
      if k_step is None:
        raise ValueError("k_step must not None when Shallow Diffusion Model inferring")
      if k_step > int(self.args.model.k_step_max):
        raise ValueError(f"k_step must <= k_step_max of Shallow Diffusion Model")
      if gt_spec is None:
        raise ValueError("gt_spec must not None when Shallow Diffusion Model inferring, gt_spec can from "
                         "input mel or output of naive model")
      print(f' [INFO] k_step_max is {self.args.model.k_step_max}.')

    aug_shift = torch.from_numpy(np.array([[float(aug_shift)]])).float().to(self.device)

    # spk_id
    spk_emb_dict = None
    if self.args.model.use_speaker_encoder:  # with speaker encoder
      spk_mix_dict, spk_emb = self.pre_spk_emb(spk_id, spk_mix_dict, len(units), spk_emb)
    # without speaker encoder
    else:
      spk_id = torch.LongTensor(np.array([[int(spk_id)]])).to(self.device)

    if k_step is not None:
      print(f' [INFO] get k_step, do shallow diffusion {k_step} steps')
    else:
      print(f' [INFO] Do full 1000 steps depth diffusion {k_step}')

    print(f" [INFO] method:{method}; infer_speedup:{infer_speedup}")
    return self.model(units, f0, volume, spk_id=spk_id, spk_mix_dict=spk_mix_dict, aug_shift=aug_shift, gt_spec=gt_spec,
                      # infer=True, infer_speedup=infer_speedup, method=method, k_step=k_step, use_tqdm=use_tqdm, spk_emb=spk_emb, spk_emb_dict=spk_emb_dict)
                      infer=True, infer_speedup=infer_speedup, method=method, k_step=k_step, use_tqdm=use_tqdm)

  @torch.no_grad()  # 比__call__多了声码器代码，输出波形
  def infer(self, units, f0, volume, gt_spec=None, spk_id=1, spk_mix_dict=None, aug_shift=0,
            infer_speedup=10, method='dpm-solver', k_step=None, use_tqdm=True, spk_emb=None):
    if k_step is not None:
      if self.naive_model is not None:
        gt_spec = self.naive_model_call(units, f0, volume, spk_id=spk_id, spk_mix_dict=spk_mix_dict,
                                        aug_shift=aug_shift, spk_emb=spk_emb)
        print(f" [INFO] get mel from naive model out.")
      assert gt_spec is not None
      if self.naive_model is None:
        raise ValueError("Please understand what you're doing")
      k_step = int(k_step)
      gt_spec = gt_spec
    else:
      gt_spec = None

    out_mel = self.__call__(units, f0, volume, spk_id=spk_id, spk_mix_dict=spk_mix_dict, aug_shift=aug_shift,
                            gt_spec=gt_spec, infer_speedup=infer_speedup, method=method, k_step=k_step,
                            use_tqdm=use_tqdm, spk_emb=spk_emb)
    return self.mel2wav(out_mel, f0)


  @torch.no_grad()  # 切片从音频推理代码
  def infer_from_long_audio(self, audio, sr=44100, key=0, spk_id=1, spk_mix_dict=None, aug_shift=0,
                            infer_speedup=10, method='dpm-solver', k_step=None, use_tqdm=True,
                            spk_emb=None,
                            threhold=-60, threhold_for_split=-40, min_len=5000, index_ratio=0):

    hop_size = self.args.data.block_size * sr / self.args.data.sampling_rate
    segments = split(audio, sr, hop_size, db_thresh=threhold_for_split, min_len=min_len)

    print(f' [INFO] Extract f0 volume and mask: Use {self.f0_model}, start...')
    _f0_start_time = time.time()
    f0 = self.extract_f0(audio, key=key, sr=sr)
    volume, mask = self.extract_volume_and_mask(audio, sr, threhold=float(threhold))
    _f0_end_time = time.time()
    _f0_used_time = _f0_end_time - _f0_start_time
    print(f' [INFO] Extract f0 volume and mask: Done. Use time:{_f0_used_time}')

    if k_step is not None:
      assert 0 < int(k_step) <= 1000
      k_step = int(k_step)
      audio_t = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
      gt_spec = self.vocoder.extract(audio_t, sr)
      gt_spec = torch.cat((gt_spec, gt_spec[:, -1:, :]), 1)
    else:
      gt_spec = None

    result = np.zeros(0)
    current_length = 0
    for segment in tqdm(segments):
      start_frame = segment[0]
      seg_input = torch.from_numpy(segment[1]).float().unsqueeze(0).to(self.device)
      seg_units = self.units_encoder.encode(seg_input, sr, hop_size)
      if index_ratio > 0:
        seg_units = self.units_indexer(units_t=seg_units, spk_id=spk_id, ratio=index_ratio)
      seg_f0 = f0[:, start_frame: start_frame + seg_units.size(1), :]
      seg_volume = volume[:, start_frame: start_frame + seg_units.size(1), :]
      if gt_spec is not None:
        seg_gt_spec = gt_spec[:, start_frame: start_frame + seg_units.size(1), :]
      else:
        seg_gt_spec = None
      seg_output = self.infer(seg_units, seg_f0, seg_volume, gt_spec=seg_gt_spec, spk_id=spk_id,
                              spk_mix_dict=spk_mix_dict,
                              aug_shift=aug_shift, infer_speedup=infer_speedup, method=method, k_step=k_step,
                              use_tqdm=use_tqdm, spk_emb=spk_emb)
      _left = start_frame * self.args.data.block_size
      _right = (start_frame + seg_units.size(1)) * self.args.data.block_size
      seg_output *= mask[:, _left:_right]
      seg_output = seg_output.squeeze().cpu().numpy()
      silent_length = round(start_frame * self.args.data.block_size) - current_length
      if silent_length >= 0:
        result = np.append(result, np.zeros(silent_length))
        result = np.append(result, seg_output)
      else:
        result = cross_fade(result, seg_output, current_length + silent_length)
      current_length = current_length + silent_length + len(seg_output)

    return result, self.args.data.sampling_rate

