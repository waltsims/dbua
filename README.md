# Differentiable Beamforming for Ultrasound Autofocusing
### [Project Page](https://www.waltersimson.com/dbua) | [Paper](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_41) | [Pre-Print](https://waltersimson.com/dbua/static/pdfs/SimsonMICCAI2023.pdf) | [Data]()


[Walter Simson](https://waltersimson.com/),
[Louise Zhuang](https://profiles.stanford.edu/louise-zhuang),
[Sergio Sanabria](https://scholar.google.es/citations?hl=es&user=E7h77bAAAAAJ),
[Neha Antil](https://med.stanford.edu/profiles/neha-antil),
[Jeremy Dahl](https://med.stanford.edu/profiles/jeremy-dahl),
[Dongwoon Hyun](https://profiles.stanford.edu/dongwoon-hyun)<br>
Stanford University

This is the official implementation of the paper "Differentiable Beamforming for Ultrasound Autofocusing."

[![dbua_video](https://img.youtube.com/vi/cUoAsEA5snE/0.jpg)](https://www.youtube.com/watch?v=cUoAsEA5snE)

## High-Level Overview

 * dbua.py - main experiment file. Adjust the global configuration parameters to run experiments.
 * das.py -  Delay-and-sum IQ data according to a given time delay profile.
 * paths.py - Calculates the time-of-flight between two points  according to a speed-of-sound map.
 * losses.py - Contains the proposed phase-error and auxiliary loss functions.

## Getting Started

This project requires ffmpeg to be installed to save mp4 files. If you have not already, install ffmpeg on Ubuntu by running:

```bash
sudo apt-get install ffmpeg
```

You can install the required Python dependencies like this:

```bash
micromamba env create -f environment.yml
micromamba activate dbua
```

Then, install JAX using the instructions for the pip GPU installation method found [here](https://jax.readthedocs.io/en/latest/installation.html#pip-installation-gpu-cuda-installed-via-pip-easier).

## Reproducing results

The hyperparameters set in dbua can be used to reproduce the results from the paper. The data from the paper can be found in the release on GitHub, and the mat files should be placed in the `data/` directory of this repository:

```
data
├── 1420.mat
├── 1465.mat
├── 1480.mat
├── 1510.mat
├── 1540.mat
├── 1555.mat
├── 1570.mat
├── checker2.mat
├── checker8.mat
├── four_layer.mat
├── inclusion_layer.mat
├── inclusion.mat
├── README.md
└── two_layer.mat
```

Run the program with this command:

```bash
python dbua.py
```

### Citation

```Bibtex
@inproceedings{simson2023dbua,
            title={Differentiable Beamforming for Ultrasound Autofocusing},
            author={Simson, Walter and Zhuang, Louise and Sanabria, Sergio J and Antil, Neha and Dahl, Jeremy J and Hyun, Dongwoon},
            booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
            pages={428--437},
            year={2023},
            organization={Springer}
          }
```

## FAQ:

- **Q:** What computer configuration was used to develop this program?
- **A:** This code was developed using the following configuration:
  
| Attribute   | Detail                                |
|-------------|---------------------------------------|
| OS          |            Ubuntu Linux               |
| RAM         | 32GB                                  |
| GPU         | NVIDIA RTX A6000  (48 GB VRAM)        |
| CUDA Version| 12.1                                  |

