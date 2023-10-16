# XLinguDial
This is the official repository for our **Short Research Paper** "Prompt Learning to Mitigate Catastrophic Forgetting in Cross-lingual Transfer for Open-domain Dialogue Generation" accepted for presentation at **the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval ([SIGIR '23](https://sigir.org/sigir2023/))**.

## Updates

July 19, 2023 - Our paper has been selected to appear in the [ACM Showcase](https://www.growkudos.com/showcase/publishers/acm?utm_medium=email&utm_source=transactional&utm_campaign=acm-n2-1) on Kudos. Check out our page [here](https://www.growkudos.com/publications/10.1145%25252F3539618.3592043/reader).

July 18, 2023 - Our paper is available on ACM Digital Library at [https://dl.acm.org/doi/10.1145/3539618.3592043](https://dl.acm.org/doi/10.1145/3539618.3592043).

May 31, 2023 - Lei Liu, the first author of this work, is honoured to receive the SIGIR Student Travel Award!

May 12, 2023 - The camera-ready paper is available on arXiv at [https://arxiv.org/abs/2305.07393](https://arxiv.org/abs/2305.07393)

## Dependencies
Our code is built upon Python 3.8 and the [🤗 Transformers version 4.8.0](https://github.com/huggingface/transformers/tree/v4.8.0). Dependencies are listed as follows.
- cuda==11.0
- torch==1.6.0
- torchvision==0.7.0
- transformers==4.8.0
- datasets==1.9.0
- arrow==2.0.0
- sacrebleu==1.5.0
- nltk
- sentencepiece
- protobuf
- absl-py
- tensorboard


## Quick Start
- Create a Python virtual environment and install all the above dependencies required by the project. Following that, activate the virtual environment.

- Download [the mT5-base model from HuggingFace 🤗Transformers](https://huggingface.co/google/mt5-base).
```
$ python download_mt5_base.py
```

- Have the data ready. Create a new folder named 'Data' in the current directory and download [the MDIA dataset](https://github.com/DoctorDream/mDIA/blob/master/datasets) to the folder. Then, the `few-shot data` in FS-XLT and `interleaved training data` in MTL can be built by running the following command. Alternatively, you can download the MDIA dataset, `few-shot data` and `interleaved training data` from [our Google Drive](https://drive.google.com/file/d/1Mv_f5EpKOU3RO-vnC3E9vlZXNqXh5b95/view?usp=sharing).
```
$ python build_data.py --directory_ori Data/MDIA/raw/train_data \
                       --directory_fs Data/MDIA_few_shot \
                       --directory_multitask Data/MDIA_multitask
```

- Turn on the offline mode for both HuggingFace 🤗 Transformers and datasets.
```
$ export HF_DATASETS_OFFLINE=1
$ export TRANSFORMERS_OFFLINE=1
```

- Run the program.

## Data

The data used for both `few-shot cross-lingual transfer learning (FS-XLT)` and `multitask learning (MTL)` in our paper are built upon [the MDIA dataset](https://github.com/DoctorDream/mDIA/blob/master/datasets), which, to the best of our knowledge, was the only publicly available multilingual benchmark for the dialogue generation task by the time we wrote our paper.

In this work, `English` is taken as the `source/auxiliary language` in FS-XLT/MTL. In terms of the `target language` in FS-XLT/MTL, we consider `Danish (da)`, `German (de)` and `Norwegian (no)` as the representatives of `Germanic language genus` along with `Spanish (es)`, `Italian (it)` and `Portuguese (pt)` as the representatives of `Romance language genus`.

##### Data for FS-XLT
(1) The `training`, `validation` and `test` data of the `source language (i.e. English)` in the `source-training stage` come from the MDIA dataset.

(2) The `few-shot data` of each `target language` in the `target-adapting stage` are randomly picked from its corresponding training set in the MDIA dataset using a fixed random seed.

(3) The `validation` and `test` data of each `target language` come from the MDIA dataset.

##### Data for MTL
(1) The `interleaved training data` in the `multitask training stage` can be built by interleaving the full `training set` of `auxiliary language (i.e. English)` with the `few-shot data` of `target language`.

(2) The `validation set` of `auxiliary language (i.e. English)` that comes from the MDIA dataset is used for model selection in the `multitask training stage`.

(3) The `test set` of each `target language` in the `target evaluation stage` comes from the MDIA dataset.


## Acknowledgments
This research is supported by the [Natural Sciences and Engineering Research Council (NSERC) of Canada](https://www.nserc-crsng.gc.ca/index_eng.asp), the York Research Chairs (YRC) program and an [ORF-RE (Ontario Research Fund Research Excellence) award](https://www.ontario.ca/page/ontario-research-fund-research-excellence) in BRAIN Alliance. Computations were made on the supercomputer [Béluga](https://www.calculquebec.ca/en/communiques/beluga-a-supercomputer-for-science-2/), managed by [Calcul Québec](https://www.calculquebec.ca/en/) and the [Digital Research Alliance of Canada](https://alliancecan.ca/en).

Lei Liu is supported by the [SIGIR Student Travel Award](https://sigir.org/general-information/travel-grants/) and [Academic Excellence Fund (AEF)](https://www.yorku.ca/gradstudies/students/current-students/awards-and-scholarships/other-funding-sources/academic-excellence-fund/) for presenting this work at SIGIR '23.

### Citation
```
@inproceedings{liu-etal-2023-prompt,
    author = {Liu, Lei and Huang, Jimmy Xiangji},
    title = {Prompt Learning to Mitigate Catastrophic Forgetting in Cross-Lingual Transfer for Open-Domain Dialogue Generation},
    year = {2023},
    isbn = {9781450394086},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3539618.3592043},
    doi = {10.1145/3539618.3592043},
    booktitle = {Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval},
    pages = {2287–2292},
    numpages = {6},
    keywords = {catastrophic forgetting, few-shot cross-lingual transfer, dialogue generation, prompt learning, multitask learning},
    location = {Taipei, Taiwan},
    series = {SIGIR '23}
}
```
