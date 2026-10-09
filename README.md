# Biomedical RCT Abstract Sentence Classification

Undergraduate final-year project at Majan University College, 2024–25, investigating sequential sentence classification of biomedical randomized controlled trial (RCT) abstracts into BACKGROUND / OBJECTIVE / METHODS / RESULTS / CONCLUSIONS. The archived TensorFlow/Keras notebooks were developed using Google Colab; a T4 GPU is recorded in the PubMed 200k notebook metadata. The experiments compare a TF-IDF baseline with token, character, hybrid and positional-feature models on PubMed 20k and 200k RCT.

> Archived original undergraduate version. A rebuilt Version 2 is in progress and will be published separately.

Implementation adapted from the SkimLit project in Daniel Bourke's TensorFlow Deep Learning course (github.com/mrdbourke/tensorflow-deep-learning). Research basis: Dernoncourt & Lee (2017), PubMed 200k RCT; Dernoncourt, Lee & Szolovits, Neural Networks for Joint Sentence Classification in Medical Paper Abstracts.

## Research basis and provenance

The research foundation is the work cited in the notebooks and thesis:

- Dernoncourt & Lee (2017), [*PubMed 200k RCT: a Dataset for Sequential Sentence Classification in Medical Abstracts*](https://arxiv.org/abs/1710.06071). This title, year and link appear in the 20k notebook.
- Dernoncourt, F., Lee, J.Y. and Szolovits, P. (2016), [*Neural networks for joint sentence classification in medical paper abstracts*](https://arxiv.org/abs/1612.05251). The thesis cites arXiv preprint arXiv:1612.05251, accessed 22 October 2024.

The dataset is [PubMed 20k/200k RCT by Franck Dernoncourt](https://github.com/Franck-Dernoncourt/pubmed-rct). It is not redistributed; the notebooks clone it at runtime. PubMed 20k is a subset of 200k, so these are not independent-corpus evaluations. Example abstracts remain in the preserved notebook outputs.

Code structure, the model sequence and much explanatory Markdown follow the [SkimLit reference implementation](https://github.com/mrdbourke/tensorflow-deep-learning/blob/main/09_SkimLit_nlp_milestone_project_2.ipynb). The notebooks fetch its `helper_functions.py` at runtime; that script is not redistributed. The upstream [MIT license](https://github.com/mrdbourke/tensorflow-deep-learning/blob/main/LICENSE) was verified on 9 October 2026 and its full notice is retained in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

### Project-specific work, i.e. not in the reference implementation

- extending the experiments to the full PubMed 200k RCT dataset, including its archive-extraction steps;
- running and comparing the model ladder on both 20k and 200k;
- a custom Universal Sentence Encoder Keras layer replacing `hub.KerasLayer`;
- diagnosing and fixing an XLA/GPU string-tensor error (`tf.config.optimizer.set_jit(False)`) and an EagerTensor→NumPy issue in the January continuation notebook;
- the requirements survey, literature review and analysis in the report.

## Interactive reading interface — added later

[**Open Paper Skimming**](https://paper-skimming-fyp.streamlit.app/) — free, public access with live historical-model inference. The free host may sleep when inactive; its first model load can take a few minutes.

[Paper Skimming](interface/README.md) is a separate reading interface added on **9 October 2026**, after the original project's completion. Paste an abstract, explore its sentence roles, inspect model scores, switch between sentence order and grouped reading, and export a readout. The interface and its pinned inference runtime do not change the archived notebooks or thesis results.

Live classification uses an existing historical tribrid checkpoint saved on 7 December 2024. Its weights stay in private storage outside this repository. Its relationship to the exact thesis runs is unverified; live predictions differ from the notebook's saved example. A separately labelled recorded example preserves the original predictions. See the [setup and deployment instructions](interface/README.md) to run it locally.

## Archived files

The December 2024 notebooks are the thesis-of-record versions. The January 2025 notebook is a post-submission continuation and is kept separate from the thesis results. The notebooks retain their original bytes, including outputs, metadata, comments and errors; only their filenames have changed. The original DOCX is retained privately. The public thesis is the PDF described below.

| Repository file | Original filename | SHA-256 |
|---|---|---|
| [notebooks/thesis_record_pubmed20k.ipynb](notebooks/thesis_record_pubmed20k.ipynb) | `AI_Deep_Learning_Machine_Learning_Natural_Language_Processing_Paper_Skimming_Model_20k_Dataset.ipynb` | `1513cfdb5cb6825acff85ce0c888ee0226cd0a7555002da9a8acbf9624ddf970` |
| [notebooks/thesis_record_pubmed200k.ipynb](notebooks/thesis_record_pubmed200k.ipynb) | `AI_Deep_Learning_Machine_Learning_Natural_Language_Processing_Paper_Skimming_Model_200k_Dataset.ipynb` | `a292d454e3689a81b2fbc87c51484509c61a10d8f0c3f1e95d3dde4ccd30b66b` |
| [notebooks/continuation_pubmed20k_jan2025.ipynb](notebooks/continuation_pubmed20k_jan2025.ipynb) | `AI_Deep_Learning_Machine_Learning_Natural_Language_Processing_Paper_Skimming_Model_20k_Dataset (1).ipynb` | `814f7f64cc37ecd209743e81ec9a50118eadfa48cb662e4bc7508b11e17d6a6c` |
| [report/final_year_report.pdf](report/final_year_report.pdf) | Public Word export of the original thesis; see modifications below | `3b5a7c07d50047d2419c9ccb29ba16c123cbfee060b3c2f963bc71626ae9fbfd` |

The public PDF removes the institutional assignment cover (including the student ID), MUC stationery/header/footer graphics, and personal document metadata. White body text was changed to black for legibility during the Word export. The retained 50 pages keep their original body text, figures, results and printed page numbers; the contents table therefore still refers to the original numbering. The unchanged source DOCX has SHA-256 `46e3061ff1a70973bc72e4e0400f225e8c8b0e1f441a964ade54c2983938e7e3` and is excluded from the public repository to keep its cover details private.

The duplicate 200k notebook, predecessor drafts, dataset files, model weights and standalone course helper script are excluded.

## Models implemented

These descriptions follow the code. Differences from the report's descriptions are documented in [ERRATA.md](ERRATA.md).

| Model | Implementation |
|---|---|
| Baseline | TF-IDF → Multinomial Naive Bayes |
| Model 1: Conv1D token | Trainable token embedding (128 dimensions) → Conv1D (64 filters, kernel size 5) → global max pooling → softmax |
| Model 2: USE | Frozen Universal Sentence Encoder from TF Hub → Dense (128) → softmax; the functional models use a custom Keras wrapper around `hub.load` |
| Model 3: Conv1D character | Character embedding (25 dimensions) → Conv1D (64 filters, kernel size 5) → global max pooling → softmax |
| Model 4: Hybrid | USE token branch + character bidirectional LSTM (25 units), concatenated → dropout / dense layers → softmax |
| Model 5: Tribrid | USE token branch + character bidirectional LSTM (32 units) + one-hot line-number and total-lines features, concatenated → dense layers → softmax; label smoothing 0.2 |

The task uses sentences from ordered abstracts, but the implementation does not jointly decode an abstract's label sequence. Model 5 supplies sentence position and abstract length as features. It is an adaptation of the reference approach, rather than an exact reproduction of the research architecture.

## Results recorded in notebook outputs

Accuracy is a percentage. F1 is scikit-learn **weighted F1**, on a 0–1 scale, as calculated by the course helper. All rows use the full official **dev** split unless marked **test**. These are preserved historical outputs, checked against the notebooks, rather than new runs.

### PubMed 20k — thesis-of-record notebook

| Model | Accuracy (%) | Weighted F1 |
|---|---:|---:|
| Baseline | 72.18 | 0.6989 |
| Conv1D token | 80.76 | 0.8059 |
| Conv1D char | 66.19 | 0.6538 |
| Hybrid | 73.65 | 0.7340 |
| Tribrid | 82.97 | 0.8286 |
| Tribrid (test) | 82.45 | 0.8233 |

### PubMed 200k — thesis-of-record notebook

| Model | Accuracy (%) | Weighted F1 |
|---|---:|---:|
| Baseline | 75.42 | 0.7440 |
| Conv1D token | 83.17 | 0.8319 |
| Conv1D char | 73.06 | 0.7287 |
| Hybrid | 77.26 | 0.7709 |
| Tribrid | 85.30 | 0.8512 |
| Tribrid (test) | 84.90 | 0.8471 |

Model 2 (Universal Sentence Encoder) produced **no recorded result in either thesis-of-record run**. The 200k notebook preserves an XLA/GPU string-tensor error; the 20k notebook has no saved training or evaluation output for that model. Model 2 scored **71.26% accuracy / 0.7095 weighted F1 on dev** only in the January continuation notebook. That later result is not part of the thesis-of-record tables above.

## Known limitations

- Neural models use `steps_per_epoch = int(0.1 * len(training_dataset))` for three epochs, with no dataset shuffling: training is heavily truncated. The baseline fits the full loaded training split. The preserved record does not establish whether successive neural-model epochs revisit the initial batches or advance through subsequent slices.
- Training-time validation also uses 10% of dev batches. The result tables come from subsequent predictions across the full dev or test split.
- Random seeds are not set. Results are single runs with no variance estimates and are not exactly reproducible.
- F1 is weighted, not macro; the tables do not establish equal performance across the five classes.
- Colab-specific paths, unpinned runtime downloads, commented extraction steps and TensorFlow/Keras compatibility issues limit fresh-session execution. The archived 200k training preview also contains numerical tokens despite its directory name indicating numbers replaced with `@`; the exact preprocessing provenance remains unresolved.

See [ERRATA.md](ERRATA.md) for implementation corrections and the limits of the historical record.

## How to run a working copy

1. Open a notebook in Google Colab and select a GPU runtime. Work in a separate Colab copy if changes are needed.
2. Run the dataset-clone and helper-download cells; internet access is required. The notebooks also load the Universal Sentence Encoder and download example abstracts at runtime.
3. For the 200k notebook, ensure `train.txt` exists before the data-reading cells. For the intended numbers-replaced split, extract its `train.zip` within `pubmed-rct/PubMed_200k_RCT_numbers_replaced_with_at_sign/`. The archived notebook's 7-Zip commands are commented out and do not provide a complete fresh-session extraction procedure.
4. Newer TensorFlow/Keras versions may need the fixes illustrated in the January continuation notebook: the custom USE layer, `tf.config.optimizer.set_jit(False)` and tensor-to-NumPy conversion. Its `.cpu()` calls already produced a deprecation warning in the saved outputs, so the continuation is not a guarantee of compatibility with current versions.

The notebooks have not been rerun for this archive. Their original runtime versions remain unpinned. Model weights and regenerated benchmark results are not supplied; the later interface has its own pinned inference dependencies.

## License

**Project code: [MIT License](LICENSE).** The course-derived material retains the upstream MIT terms and copyright notice reproduced in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

**Report/thesis: all rights reserved.** The report is excluded from any code license unless the author explicitly changes this decision. Dataset and other third-party materials retain their own terms.
