<div align="center">

## Adaptive Neighbourhoods for the Discovery<br/> of Adversarial Examples

Python API for generating adapted and unique neighbourhoods for
searching for adversarial examples

[![PyPI](https://img.shields.io/pypi/v/adaptive-neighbourhoods?style=flat-square&color=green)](https://pypi.python.org/pypi/adaptive-neighbourhoods/)
[![GitHub license](https://img.shields.io/github/license/jaypmorgan/adaptive-neighbourhoods.svg?style=flat-square)](https://github.com/jaypmorgan/adaptive-neighbourhoods/blob/master/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/adaptive-neighbourhoods/badge/?version=latest&style=flat-square)](https://adaptive-neighbourhoods.readthedocs.io/en/latest/?badge=latest)

</div>

## Installation & usage

This work is released on PyPi. Installation, therefore, is as simple as installing the package with pip:

```bash
python3 -m pip install adaptive-neighbourhoods
```

At this point, you're free to start generating neighbourhoods for your own dataset:

```python
from adaptive_neighbourhoods import epsilon_expand

neighbourhoods = epsilon_expand(
    x,  # your input data
    y)  # the integer encoded labels for your data
```

Move information on the variable parameters and general guidance on using this package can be found at: https://jaypmorgan.github.io/adaptive-neighbourhoods/

## Contributing

All contributions and feedback are welcome!

There are three main remote mirrors used for hosting this project. If
you would like to contribute, please submit an
issue/pull-request/patch-request to any of these mirrors:

- Github: https://github.com/jaypmorgan/adaptive-neighbourhoods
- Gitlab: https://gitlab.com/jaymorgan/adaptive-neighbourhoods
- Source Hut: https://git.sr.ht/~jaymorgan/adaptive-neighbourhoods

## Citing this work

If you use this work in your research, please consider referencing our
article using the following bibtex entry:

```
@article{DBLP:journals/corr/abs-2101-09108,
  author    = {Jay Morgan and
               Adeline Paiement and
               Arno Pauly and
               Monika Seisenberger},
  title     = {Adaptive Neighbourhoods for the Discovery of Adversarial Examples},
  journal   = {CoRR},
  volume    = {abs/2101.09108},
  year      = {2021},
  url       = {https://arxiv.org/abs/2101.09108},
  eprinttype = {arXiv},
  eprint    = {2101.09108},
  timestamp = {Sat, 30 Jan 2021 18:02:51 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2101-09108.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
