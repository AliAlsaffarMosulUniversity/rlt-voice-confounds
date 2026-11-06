"""Acoustic feature extraction replicating the reviewed paper's 5 feature families (Praat/Parselmouth + librosa),
plus a separate 'channel' feature set computed only from the quietest (non-speech) frames."""
import os, sys
os.chdir(os.environ.get('RLT_WORK', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'work')))

import numpy as np, pandas as pd, parselmouth, librosa, warnings, sys
from parselmouth.praat import call
from multiprocessing import Pool
warnings.filterwarnings("ignore")
W = "wav/"

def feats(fid):
    y, sr = librosa.load(W + "trial_" + fid + ".wav", sr=16000)
    yt, _ = librosa.effects.trim(y, top_db=25)
    f = {"id": fid}
    snd = parselmouth.Sound(yt, sampling_frequency=sr)
    # --- prosodic (gender-neutral pitch range 75-500 Hz)
    pitch = snd.to_pitch(pitch_floor=75, pitch_ceiling=500)
    f0 = pitch.selected_array["frequency"]; v = f0[f0 > 0]
    f["f0_mean"] = v.mean() if len(v) else np.nan
    f["f0_std"] = v.std() if len(v) else np.nan
    f["f0_range"] = (np.percentile(v, 95) - np.percentile(v, 5)) if len(v) else np.nan
    f["voiced_ratio"] = len(v) / max(1, len(f0))
    # --- voice quality
    pp = call(snd, "To PointProcess (periodic, cc)", 75, 500)
    f["jitter_local"] = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    f["jitter_rap"] = call(pp, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    f["shimmer_local"] = call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    f["shimmer_apq3"] = call([snd, pp], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    harm = snd.to_harmonicity_cc(); f["hnr"] = call(harm, "Get mean", 0, 0)
    # --- intensity / energy
    inten = snd.to_intensity(); iv = inten.values.ravel()
    f["int_mean"] = iv.mean(); f["int_std"] = iv.std()
    rms = librosa.feature.rms(y=yt)[0]; f["rms_mean"] = rms.mean(); f["rms_std"] = rms.std()
    # --- temporal
    iv_ = librosa.effects.split(yt, top_db=30)
    sp = sum(e - s for s, e in iv_) / sr; dur = len(yt) / sr
    f["dur"] = dur; f["speech_ratio"] = sp / dur
    gaps = [(iv_[i + 1][0] - iv_[i][1]) / sr for i in range(len(iv_) - 1)]
    f["n_pauses"] = sum(g > 0.25 for g in gaps); f["pause_rate"] = f["n_pauses"] / dur
    # --- spectral
    mf = librosa.feature.mfcc(y=yt, sr=sr, n_mfcc=13)
    for i in range(13):
        f[f"mfcc{i+1}_mean"] = mf[i].mean(); f[f"mfcc{i+1}_std"] = mf[i].std()
    f["spec_centroid"] = librosa.feature.spectral_centroid(y=yt, sr=sr).mean()
    f["spec_bandwidth"] = librosa.feature.spectral_bandwidth(y=yt, sr=sr).mean()
    f["spec_rolloff"] = librosa.feature.spectral_rolloff(y=yt, sr=sr).mean()
    f["zcr"] = librosa.feature.zero_crossing_rate(yt).mean()
    # --- channel-only features: quietest 15% of frames of the untrimmed signal (background / room / codec)
    S = np.abs(librosa.stft(y, n_fft=512, hop_length=160)); e = S.sum(0)
    q = S[:, e <= np.percentile(e, 15)]
    lq = np.log(q + 1e-8); bands = np.array_split(np.arange(lq.shape[0]), 8)
    for j, b in enumerate(bands): f[f"ch_band{j}"] = lq[b].mean()
    f["ch_level"] = np.log(q.mean() + 1e-8)
    f["ch_flat"] = np.exp(lq.mean(0)).mean() / (q.mean(0).mean() + 1e-12)
    return f

if __name__ == "__main__":
    ids = pd.read_csv("meta.csv").id.tolist()
    with Pool(2) as P: rows = P.map(feats, ids)
    pd.DataFrame(rows).to_csv("features.csv", index=False); print("done", len(rows))
