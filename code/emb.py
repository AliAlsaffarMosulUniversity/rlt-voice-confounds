import os, sys
os.chdir(os.environ.get('RLT_WORK', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'work')))
import numpy as np, pandas as pd, torch
torch.set_num_threads(1)
from resemblyzer import VoiceEncoder, preprocess_wav
enc=VoiceEncoder('cpu'); ids=pd.read_csv('meta.csv').id.tolist(); rows=[]
for i in ids:
    w=preprocess_wav(f'wav/trial_{i}.wav'); e=enc.embed_utterance(w); rows.append([i]+list(e))
pd.DataFrame(rows,columns=['id']+[f'e{k}' for k in range(256)]).to_csv('resemblyzer.csv',index=False); print('done',len(rows))
