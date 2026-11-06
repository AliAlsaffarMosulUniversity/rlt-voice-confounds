# Does the Voice Reveal the Lie or the Liar?

Speaker, gender, role and recording confounds in acoustic deception detection on the **Real-Life Trial (RLT)** dataset.

Code, anonymised annotations and full results for the paper:

> Alsaffar, A. A. (2026). *Does the Voice Reveal the Lie or the Liar? Speaker, Gender, Role and Recording Confounds in Acoustic Deception Detection on the Real-Life Trial Dataset.* Manuscript under review.

## Key findings

| Setting | Best accuracy |
|---|---|
| Random 10-fold clips (common practice) | 75–76% |
| Speaker-grouped folds, 47 speech features (SVM) | 56.1% |
| Trial-grouped folds, 47 speech features (SVM) | 49.5% |
| Grouped folds, any representation, after within-gender normalisation | 40.6–52.0% |
| Rule: *witness → truthful, otherwise deceptive* (no audio) | **77.7%** |
| Rule: *female → deceptive* (no audio) | 68.6% |

- The same acoustic features identify the speaker of a held-out clip with 90.0% accuracy (98.8% with a pretrained GE2E speaker encoder).
- Features from non-speech (background) frames alone identify the source trial with 73.8% accuracy.
- eGeMAPS adds ranking information beyond gender on all clips (conditional permutation AUC p = 0.010–0.035), but none among non-witness clips (p ≥ 0.40): the residual signal is witness role.
- Label-permutation tests are significant under random folds (p = 0.007) but not under speaker- or trial-grouped folds after gender control (p ≥ 0.47).

## Repository layout

```
annotations/meta.csv   per-clip label, role, anonymised trial (T01–T32) and speaker (S01–S56) codes, perceived gender
code/                  feature extraction, experiments, statistics, figures
results/               per-repetition CV results, permutation results, extracted features
figures/               paper figures
```

`meta.csv` also corrects a duplicated entry in the dataset README (`trial_truth_008` is listed twice; the second entry is `trial_truth_009`).

## Data

The audio/video are **not** included. Obtain the *Real-Life Trial Deception Detection Dataset* v1.0 (research use only) from the University of Michigan LIT lab: <https://lit.eecs.umich.edu/downloads.html> and cite Pérez-Rosas et al. (2015).

## Reproducing

```bash
pip install -r requirements.txt
mkdir work && cp annotations/meta.csv work/
# extract 16 kHz mono audio for every clip into work/wav/
for f in Real-life_Deception_Detection_2016/Clips/*/*.mp4; do
  ffmpeg -loglevel error -i "$f" -vn -ac 1 -ar 16000 "work/wav/$(basename "${f%.mp4}").wav"; done

python code/features.py        # 47 speech + 10 channel features  -> work/features.csv
python code/smile.py           # eGeMAPS, ComParE-2016            -> work/egemaps.csv, work/compare.csv
python code/emb.py             # Resemblyzer GE2E embeddings      -> work/resemblyzer.csv
python code/experiments.py     # 5 classifiers x 3 protocols x 10 repeats
python code/baselines.py       # rules and leakage probes
python code/validate_annot.py resemblyzer   # annotation validation
python code/perm.py 150        # permutation tests
python code/subsets.py
python code/extended.py && python code/extended2.py
python code/incremental.py && python code/incremental_role.py   # eGeMAPS beyond gender / role
python code/analysis.py        # bootstrap CIs and figures
```

Scripts run in `work/` by default (override with `RLT_WORK=/path`). Speaker codes preserve the sort order used in the paper, so fold assignments and numbers reproduce exactly.

## Reporting checklist for deception-detection studies on small multi-clip datasets

1. Report speaker-grouped (and, if possible, session/trial-grouped) results, with grouped inner model selection.
2. Report majority, gender and role baselines under the same protocol.
3. Report the number of speakers and clips per speaker in each class.
4. Use speaker-level bootstrap intervals and permutation tests under the grouped protocol.
5. Include a leakage probe (e.g. speaker identification from the same features).
6. Release code and the speaker grouping used.

## Ethics

Gender labels are perceived binary labels used only to test for bias. Speaker identities were annotated only to define cross-validation groups and are released as anonymous codes. We caution against operational use of automated deception detection trained on datasets of this kind.

## License

Code: MIT. Annotations and results: CC BY 4.0.

## Contact

Ali A. Alsaffar — College of Administration and Economics, University of Mosul, Iraq · ali_alsaffar@uomosul.edu.iq · ORCID [0009-0005-0918-893X](https://orcid.org/0009-0005-0918-893X)
