# XLinguDial
This is the official repository for our **Short Research Paper** "Prompt Learning to Mitigate Catastrophic Forgetting in Cross-lingual Transfer for Open-domain Dialogue Generation" accepted for presentation at **the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval ([SIGIR '23](https://sigir.org/sigir2023/))**.

## Updates

July 18, 2023 - Our paper is available on ACM Digital Library at [https://dl.acm.org/doi/10.1145/3539618.3592043](https://dl.acm.org/doi/10.1145/3539618.3592043).

June 12, 2023 - We are working on cleaning up our code and will make it available soon.

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
