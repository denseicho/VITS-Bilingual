import os
import sys
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

from scipy.io.wavfile import write

def get_text(text, lang, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners, lang)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


hps_ms = utils.get_hparams_from_file("./configs/chipanese_base.json")

net_g_ms = SynthesizerTrn(
    len(symbols),
    hps_ms.data.filter_length // 2 + 1,
    hps_ms.train.segment_size // hps_ms.data.hop_length,
    n_speakers=hps_ms.data.n_speakers,
    **hps_ms.model)
_ = net_g_ms.eval()

_ = utils.load_checkpoint('generator_chipanese.pth', net_g_ms, None)

print("----------param---------- ")
print("argv1", sys.argv[1])
print("----------param---------- ")

filename = sys.argv[1];
language = "Japanese"
speaker_id = "0" #@param [0, 1]
length_scale = 1 #@param {type:"slider", min:0.1, max:3, step:0.05}

if language == 'Chinese':
  lang = 'ch'
elif language == 'Japanese':
  lang = 'jp'

speaker_id = int(speaker_id)
sid = torch.LongTensor([speaker_id])

f = open(filename, 'r', encoding='UTF-8')
datalist = f.readlines()
for data in datalist:
  print(data)
  strList = data.split(':')
  text = strList[1]
  print('text: ' + text)
  stn_tst = get_text(text, lang, hps_ms)
  with torch.no_grad():
      x_tst = stn_tst.unsqueeze(0)
      x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
      audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=length_scale)[0][0,0].data.float().numpy()

  # Save it on the disk
  voicefilename = './voice/' + strList[0]
  audio_path = f'{voicefilename}.wav'
  print('audio_path: ' + audio_path)
  write(audio_path, 22050, audio)
  print(audio_path + ' saved.')

f.close()