# Towards Fair Graph Anomaly Detection: Problem, New Datasets, and Evaluation

_**NEW!** Accepted at CIKM 2024._

Our datasets (Reddit, Twitter) are publicly available through [this link](https://www.dropbox.com/scl/fi/5vga0qe9bqdmwroz7uefc/FairGAD-datasets.tar.xz?rlkey=1rmkp34xovis7xtph216xysgl&dl=0) ([alt link](https://www.dropbox.com/scl/fi/cul06biq4x1h4quwwrt8g/FairGAD-datasets.tar.xz?rlkey=tbcjp26s8bzg2bi0rkg94n10e&st=kyw80sgx&dl=0)) as PyTorch Geometric datasets. 

## Environment
  - python=3.8
  - pytorch
  - pyg
  - networkx
  - scipy

Please note that our work builds upon [PyGOD](https://github.com/pygod-team/pygod). However, **our work uses version 0.3** due to the different implementation of CoLA. In PyGOD v1.1 as of May 2024, [their implementation](https://github.com/pygod-team/pygod/blob/main/pygod/detector/cola.py) of CoLA is different due to the use of neighbour sampling, instead of random walk sampling that is present in v0.3 and as described in the CoLA paper. We hope this clarifies why an older version of PyGOD is included in our repository. As a result, we have included the PyGOD v0.3 repo here. 

## Our implementations

We implemented the fairness regularisers (FairOD, HIN, correlation) in the `utils.py` file that takes in the model’s raw anomaly score or loss and the sensitive attributes to calculate a fairness loss.
We also implemented the ADCG regulariser (to improve equality of odds) in the same `utils.py` file. 

We have modified the files for DOMINANT, CONAD, CoLA, DONE, AdONE, and included a new file for VGOD to include fairness regularizers. Please see the correspondig `fit_with_fairness()` method that calls the above fairness regularisers while training the model. If the ADCG regulariser is used, the ideal DCG score is also calculated in this method. 

## Sample runs

Please see `fairGAD/test_fair_fitting.py`. This file is our main driver file to obtain the results for all tests used in this paper. 

## User Account Inquiry and Removal

Please contact `n​nn​k [at] gatec​h [dot] e​du` with an email titled "FairGAD - Account Inquiry and Removal" including the your username and platform (Reddit/Twitter) to check if your account is used in our dataset and wish to be removed from it. 

## Data Use Agreement

If you would like to use our dataset, please contact the same email address above with an email titled "FairGAD - Data Use Agreement" and we will get in touch with you. 

## Potential Twitter bots

We have included a list of node indexes that correspond to Twitter accounts that have a Botometer "raw_overall" score of greater than 0.9 that may be possible Twitter bot accounts. 

## Citing FairGAD

Accepted at CIKM 2024. We would appreciate a citation to the following paper if you have used our work:

    @inproceedings{neoFairGraphAnomaly2024a,
      title = {Towards {{Fair Graph Anomaly Detection}}: {{Problem}}, {{Benchmark Datasets}}, and {{Evaluation}}},
      shorttitle = {Towards {{Fair Graph Anomaly Detection}}},
      booktitle = {Proceedings of the 33rd {{ACM International Conference}} on {{Information}} and {{Knowledge Management}}},
      author = {Neo, Neng Kai Nigel and Lee, Yeon-Chang and Jin, Yiqiao and Kim, Sang-Wook and Kumar, Srijan},
      date = {2024-10-21},
      pages = {1752--1762},
      publisher = {ACM},
      location = {Boise ID USA},
      doi = {10.1145/3627673.3679754},
      url = {https://dl.acm.org/doi/10.1145/3627673.3679754},
      urldate = {2024-11-24},
      eventtitle = {{{CIKM}} '24: {{The}} 33rd {{ACM International Conference}} on {{Information}} and {{Knowledge Management}}},
      isbn = {9798400704369},
      langid = {english}
    }
