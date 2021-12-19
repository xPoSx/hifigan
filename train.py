import os

import librosa
import torch
from torch import nn
from itertools import chain
import random
from tqdm import tqdm
from src.model import Generator, MPD, MSD
from torch.utils.data import DataLoader
from src.dataset import LJSpeechDataset
from src.collate import LJSpeechCollator
from src.featurizer import MelSpectrogram, MelSpectrogramConfig
from src.loss import GenLoss, DiscLoss, FeatureLoss
from src.logger import WanDBWriter

bs = 16

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = WanDBWriter()
featurizer = MelSpectrogram(MelSpectrogramConfig()).to(device)
dataloader = DataLoader(LJSpeechDataset('.'), batch_size=bs, collate_fn=LJSpeechCollator())
gen = Generator(80).to(device)
mpd = MPD().to(device)
msd = MSD().to(device)

mel_loss = nn.L1Loss()
feat_loss = FeatureLoss()
gen_loss = GenLoss()
disc_loss = DiscLoss()

gen_opt = torch.optim.AdamW(gen.parameters(), lr=2e-4, betas=(0.8, 0.99))
disc_opt = torch.optim.AdamW(chain(mpd.parameters(), msd.parameters()), lr=2e-4, betas=(0.8, 0.99))

gen_scheduler = torch.optim.lr_scheduler.ExponentialLR(gen_opt, 0.999)
disc_scheduler = torch.optim.lr_scheduler.ExponentialLR(disc_opt, 0.999)

test_wavs = os.listdir('test_wavs')

test_ts = [
    'A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest',
    'Massachusetts Institute of Technology may be best known for its math, science and engineering education',
    'Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space',
]

gen.train()
mpd.train()
msd.train()

try:
    gen.load_state_dict(torch.load('hifigan'))
except:
    pass

n_epochs = 200
segment_size = 8192

for e in range(n_epochs):
    for i, batch in tqdm(enumerate(dataloader)):
        waveform = batch.waveform.to(device)

        segment_start = random.randint(0, max(waveform.shape[1] - segment_size, 0))
        waveform = waveform[:, segment_start:segment_start + segment_size]

        mels = featurizer(waveform)
        waveform = waveform.unsqueeze(1)
        gen_wav = gen(mels)
        gen_mel = featurizer(gen_wav).squeeze(1)

        disc_opt.zero_grad()
        mpd_true_outs, _, mpd_pred_outs, _ = mpd(waveform, gen_wav.detach())
        mpd_loss = disc_loss(mpd_true_outs, mpd_pred_outs)

        msd_true_outs, _, msd_pred_outs, _ = msd(waveform, gen_wav.detach())
        msd_loss = disc_loss(msd_true_outs, msd_pred_outs)

        discriminator_loss = mpd_loss + msd_loss
        discriminator_loss.backward()
        disc_opt.step()

        gen_opt.zero_grad()
        mels_loss = mel_loss(mels, gen_mel) * 45

        _, mpd_true_feats, mpd_pred_outs, mpd_pred_feats = mpd(waveform, gen_wav)
        _, msd_true_feats, msd_pred_outs, msd_pred_feats = msd(waveform, gen_wav)

        mpd_feat_loss = feat_loss(mpd_true_feats, mpd_pred_feats)
        msd_feat_loss = feat_loss(msd_true_feats, msd_pred_feats)
        mpd_gen_loss = gen_loss(mpd_pred_outs)
        msd_gen_loss = gen_loss(msd_pred_outs)

        generator_loss = mels_loss + mpd_gen_loss + msd_gen_loss + (mpd_feat_loss + msd_feat_loss) * 2

        generator_loss.backward()
        gen_opt.step()
        logger.add_metrics({"Gen loss": generator_loss.item(), "Disc loss": discriminator_loss.item()})
        if i % 200 == 0:
            gen.eval()
            with torch.no_grad():
                for j, filename in enumerate(test_wavs):
                    test_wave, _ = librosa.load(os.path.join('test_wavs', filename), sr=22050)
                    test_wave = torch.Tensor(test_wave).unsqueeze(0).to(device)
                    mels = featurizer(test_wave)
                    pred = gen(mels)
                    pred_mels = featurizer(pred).squeeze(1)
                    logger.add_audio(pred.cpu(), test_wave.cpu(), test_ts[j], j)
                    logger.add_spectrogram(pred_mels.cpu(), mels.cpu(), test_ts[j], j)
            gen.train()

    gen_scheduler.step()
    disc_scheduler.step()
    logger.add_metrics({"Learning rate": gen_scheduler.get_last_lr()[0]})
    torch.save(gen.state_dict(), 'hifigan')
