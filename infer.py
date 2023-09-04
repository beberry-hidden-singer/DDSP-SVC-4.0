"""modified from https://github.com/yxlllc/DDSP-SVC/blob/master/main_diff.py
by karljeon44"""
import argparse
import hashlib
import logging
logging.getLogger("fairseq").setLevel(logging.WARNING)
import os
from ast import literal_eval

import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from ddsp_naive.core import upsample
from ddsp_naive.vocoder import F0Extractor, UnitsEncoder, VolumeExtractor
from diffusion.vocoder import load_model_vocoder
from tools.infer_tools import DiffusionSVC
from tools.slicer import cross_fade, split


def check_args(ddsp_args, diff_args):
  if ddsp_args.data.sampling_rate != diff_args.data.sampling_rate:
    print("Unmatch data.sampling_rate!")
    return False
  if ddsp_args.data.block_size != diff_args.data.block_size:
    print("Unmatch data.block_size!")
    return False
  if ddsp_args.data.encoder != diff_args.data.encoder:
    print("Unmatch data.encoder!")
    return False
  return True

def parse_args(args=None, namespace=None):
  """Parse command-line arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "-ddsp",
    "--ddsp_ckpt",
    type=str,
    required=False,
    default=None,
    help="path to the DDSP model checkpoint (for shallow diffusion)",
  )
  parser.add_argument(
    "-nmodel",
    "--nmodel_ckpt",
    type=str,
    required=False,
    default=None,
    help="path to the naive + shallow diffusion combo checkpoint (for shallow diffusion)",
  )
  parser.add_argument(
    "-d",
    "--device",
    type=str,
    default=None,
    required=False,
    help="cpu or cuda, auto if not set")
  parser.add_argument(
    "-i",
    "--input",
    type=str,
    required=True,
    help="path to the input audio file",
  )
  parser.add_argument(
    "-o",
    "--output",
    type=str,
    required=True,
    help="path to the output audio file",
  )
  parser.add_argument(
    "-id",
    "--spk_id",
    type=str,
    required=False,
    default=1,
    help="speaker id (for multi-speaker model) | default: 1",
  )
  parser.add_argument(
    "-mix",
    "--spk_mix_dict",
    type=str,
    required=False,
    default="None",
    help="mix-speaker dictionary (for multi-speaker model) | default: None",
  )
  parser.add_argument(
    "-k",
    "--key",
    type=str,
    required=False,
    default=0,
    help="key changed (number of semitones) | default: 0",
  )
  parser.add_argument(
    "-f",
    "--formant_shift_key",
    type=str,
    required=False,
    default=0,
    help="formant changed (number of semitones) , only for pitch-augmented model| default: 0",
  )
  parser.add_argument(
    "-pe",
    "--pitch_extractor",
    type=str,
    required=False,
    default='rmvpe',
    help="pitch extrator type: parselmouth, dio, harvest, crepe, rmvpe (default)",
  )
  parser.add_argument(
    "-fmin",
    "--f0_min",
    type=str,
    required=False,
    default=50,
    help="min f0 (Hz) | default: 50",
  )
  parser.add_argument(
    "-fmax",
    "--f0_max",
    type=str,
    required=False,
    default=1100,
    help="max f0 (Hz) | default: 1100",
  )
  parser.add_argument(
    "-th",
    "--threhold",
    type=str,
    required=False,
    default=-60,
    help="response threhold (dB) | default: -60",
  )
  parser.add_argument(
    "-th4sli",
    "--threhold_for_split",
    type=str,
    required=False,
    default=-40,
    help="threhold for split (dB) | default: -40",
  )
  parser.add_argument(
    "-min_len",
    "--min_len",
    type=str,
    required=False,
    default=5000,
    help="min split len | default: 5000",
  )
  parser.add_argument(
    "-diffid",
    "--diff_spk_id",
    type=str,
    required=False,
    default='auto',
    help="diffusion speaker id (for multi-speaker model) | default: auto",
  )
  parser.add_argument(
    "-speedup",
    "--speedup",
    type=str,
    required=False,
    default='auto',
    help="speed up | default: auto",
  )
  parser.add_argument(
    "-method",
    "--method",
    type=str,
    required=False,
    default='auto',
    help="ddim, pndm, dpm-solver or unipc | default: auto",
  )
  parser.add_argument(
    "-kstep",
    "--kstep",
    type=str,
    required=False,
    default=None,
    help="shallow diffusion steps | default: None",
  )
  parser.add_argument(
    "-ir",
    "--index_ratio",
    type=str,
    required=False,
    default=0,
    help="index_ratio, if > 0 will use index | default: 0",
  )
  return parser.parse_args(args=args, namespace=namespace)


if __name__ == '__main__':
  # parse commands
  cmd = parse_args()

  device = cmd.device
  if device is None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

  # load diffusion model
  assert cmd.ddsp_ckpt is not None or cmd.nmodel_ckpt is not None

  # Naive + Shallow Diffusion Inference
  if cmd.nmodel_ckpt is not None:
    diffusion_svc = DiffusionSVC(device=device)  # 加载模型
    diffusion_svc.load_model(model_path=cmd.nmodel_ckpt, f0_model=cmd.pitch_extractor, f0_max=cmd.f0_max, f0_min=cmd.f0_min)

    spk_mix_dict = literal_eval(cmd.spk_mix_dict)

    # load wav
    in_wav, in_sr = librosa.load(cmd.input, sr=None)
    if len(in_wav.shape) > 1:
      in_wav = librosa.to_mono(in_wav)

    if cmd.speedup == 'auto':
      cmd.speedup = 10

    if cmd.kstep is None:
      print("Defaulting to 100 kstep")
      cmd.kstep = 100

    if cmd.method == 'auto':
      cmd.method = 'dpm-solver'

    if cmd.speedup > 1:
      print('Sampling method: '+ cmd.method)
      print('Speed up: '+ str(cmd.speedup))
    else:
      cmd.method = 'dpm-solver'
      print('Sampling method: DDPM')

    # infer
    result, out_sr = diffusion_svc.infer_from_long_audio(
      in_wav, sr=in_sr,
      key=float(cmd.key),
      spk_id=int(cmd.spk_id),
      spk_mix_dict=spk_mix_dict,
      aug_shift=int(cmd.formant_shift_key),
      infer_speedup=int(cmd.speedup),
      method=cmd.method,
      k_step=cmd.kstep,
      use_tqdm=True,
      spk_emb=None,
      threhold=float(cmd.threhold),
      threhold_for_split=float(cmd.threhold_for_split),
      min_len=int(cmd.min_len),
      index_ratio=float(cmd.index_ratio)
    )

    # save
    sf.write(cmd.output, result, out_sr)

  else: # DDSP + Shallow Diffusion Inference
    model, vocoder, args = load_model_vocoder(cmd.ddsp_ckpt, device=device)

    # load input
    audio, sample_rate = librosa.load(cmd.input, sr=None)
    if len(audio.shape) > 1:
      audio = librosa.to_mono(audio)
    hop_size = args.data.block_size * sample_rate / args.data.sampling_rate

    # get MD5 hash from wav file
    md5_hash = ""
    with open(cmd.input, 'rb') as f:
      data = f.read()
      md5_hash = hashlib.md5(data).hexdigest()
      print("MD5: " + md5_hash)

    cache_dir_path = os.path.join(os.path.dirname(__file__), "cache")
    cache_file_path = os.path.join(cache_dir_path, f"{cmd.pitch_extractor}_{hop_size}_{cmd.f0_min}_{cmd.f0_max}_{md5_hash}.npy")

    is_cache_available = os.path.exists(cache_file_path)
    if is_cache_available:
      # f0 cache load
      print('Loading pitch curves for input audio from cache directory...')
      f0 = np.load(cache_file_path, allow_pickle=False)
    else:
      # extract f0
      print('Pitch extractor type: ' + cmd.pitch_extractor)
      pitch_extractor = F0Extractor(
        cmd.pitch_extractor,
        sample_rate,
        hop_size,
        float(cmd.f0_min),
        float(cmd.f0_max))
      print('Extracting the pitch curve of the input audio...')
      f0 = pitch_extractor.extract(audio, uv_interp = True, device = device)

      # f0 cache save
      os.makedirs(cache_dir_path, exist_ok=True)
      np.save(cache_file_path, f0, allow_pickle=False)

    f0 = torch.from_numpy(f0).float().to(device).unsqueeze(-1).unsqueeze(0)

    # key change
    f0 = f0 * 2 ** (float(cmd.key) / 12)

    # formant change
    formant_shift_key = torch.from_numpy(np.array([[float(cmd.formant_shift_key)]])).float().to(device)

    # extract volume
    print('Extracting the volume envelope of the input audio...')
    volume_extractor = VolumeExtractor(hop_size)
    volume = volume_extractor.extract(audio)
    # mask = (volume > 10 ** (float(cmd.threhold) / 20)).astype('float')
    # mask = np.pad(mask, (4, 4), constant_values=(mask[0], mask[-1]))
    # mask = np.array([np.max(mask[n : n + 9]) for n in range(len(mask) - 8)])
    # mask = torch.from_numpy(mask).float().to(device).unsqueeze(-1).unsqueeze(0)
    # mask = upsample(mask, args.data.block_size).squeeze(-1)
    mask = volume_extractor.get_mask_from_volume(volume, args.data.block_size, cmd.threhold, device)
    volume = torch.from_numpy(volume).float().to(device).unsqueeze(-1).unsqueeze(0)

    # load units encoder
    if args.data.encoder == 'cnhubertsoftfish':
      cnhubertsoft_gate = args.data.cnhubertsoft_gate
    else:
      cnhubertsoft_gate = 10
    units_encoder = UnitsEncoder(
      args.data.encoder,
      args.data.encoder_ckpt,
      args.data.encoder_sample_rate,
      args.data.encoder_hop_size,
      cnhubertsoft_gate=cnhubertsoft_gate,
      device = device
    )

    # speaker id or mix-speaker dictionary
    spk_mix_dict = literal_eval(cmd.spk_mix_dict)
    spk_id = torch.LongTensor(np.array([[int(cmd.spk_id)]])).to(device)
    if cmd.diff_spk_id == 'auto':
      diff_spk_id = spk_id
    else:
      diff_spk_id = torch.LongTensor(np.array([[int(cmd.diff_spk_id)]])).to(device)

    if spk_mix_dict is not None:
      print('Mix-speaker mode')
    else:
      print('DDSP Speaker ID: '+ str(int(cmd.spk_id)))
      print('Diffusion Speaker ID: '+ str(cmd.diff_spk_id))

    # speed up
    if cmd.speedup == 'auto':
      infer_speedup = args.infer.speedup
    else:
      infer_speedup = int(cmd.speedup)

    if cmd.method == 'auto':
      method = args.infer.method
    else:
      method = cmd.method

    if infer_speedup > 1:
      print('Sampling method: '+ method)
      print('Speed up: '+ str(infer_speedup))
    else:
      print('Sampling method: DDPM')

    assert args.model.type == 'DDSP'

    input_mel = k_step = None
    if cmd.kstep is not None:
      k_step = int(cmd.kstep)
      if k_step > args.model.k_step_max:
        k_step = args.model.k_step_max
    else:
      k_step = args.model.k_step_max
    print('Shallow diffusion step: ' + str(k_step))

    # forward and save the output
    result = np.zeros(0)
    current_length = 0
    segments = split(audio, sample_rate, hop_size)
    print('Cut the input audio into ' + str(len(segments)) + ' slices')
    with torch.no_grad():
      for segment in tqdm(segments):
        start_frame = segment[0]
        seg_input = torch.from_numpy(segment[1]).float().unsqueeze(0).to(device)
        seg_units = units_encoder.encode(seg_input, sample_rate, hop_size)

        seg_f0 = f0[:, start_frame : start_frame + seg_units.size(1), :]
        seg_volume = volume[:, start_frame : start_frame + seg_units.size(1), :]
        if input_mel is not None:
          seg_input_mel = input_mel[:, start_frame : start_frame + seg_units.size(1), :]
        else:
          seg_input_mel = None

        seg_mel = model(
          seg_units,
          seg_f0,
          seg_volume,
          spk_id = diff_spk_id,
          spk_mix_dict = spk_mix_dict,
          aug_shift = formant_shift_key,
          vocoder=vocoder,
          gt_spec=seg_input_mel,
          infer=True,
          infer_speedup=infer_speedup,
          method=method,
          k_step=k_step)
        seg_output = vocoder.infer(seg_mel, seg_f0)
        seg_output *= mask[:, start_frame * args.data.block_size : (start_frame + seg_units.size(1)) * args.data.block_size]
        seg_output = seg_output.squeeze().cpu().numpy()

        silent_length = round(start_frame * args.data.block_size) - current_length
        if silent_length >= 0:
          result = np.append(result, np.zeros(silent_length))
          result = np.append(result, seg_output)
        else:
          result = cross_fade(result, seg_output, current_length + silent_length)
        current_length = current_length + silent_length + len(seg_output)

    # finally write
    sf.write(cmd.output, result, args.data.sampling_rate)
