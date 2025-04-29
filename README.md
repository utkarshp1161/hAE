# Human-in-the-Loop: Automated Experiments (hAE)
[![DOI](https://zenodo.org/badge/777496073.svg)](https://doi.org/10.5281/zenodo.15175786)

## Paper : [Building Workflows for Interactive Human in the LoopAutomated Experiment (hAE) in STEM-EELS](https://pubs.rsc.org/en/content/articlelanding/2025/dd/d5dd00033e)

![Image](assets/AE.jpg)


## Description

The concept of Human-in-the-Loop for Automated Experiments (hAE) represents an innovative approach to scientific research, blending human expertise with automated systems to optimize experimental processes. 

## How to run:
- clone the repository:
- cd hAE
- pip install .
- pip install atomai
- pip install pyroved
- try example data: !wget https://www.dropbox.com/s/nicyvttpihzh1cd/Plasmonic_sets_7222021_fixed.npy [update file path in example.yaml]
- python scripts/active_learning.py --config configs/example.yaml
- python scripts/full_dkl_benchmark.py --config configs/example.yaml
- python scripts/forensics.py --config configs/example.yaml
 
## coming soon: intervention notebooks

## Credits and References


- AtomAI:  [GitHub Repository](https://github.com/pycroscopy/atomai)
  
- AE-DKL:  [GitHub Repository](https://github.com/kevinroccapriore/AE-DKL)
  
- AE-PostExperimentAnalysis-DKL-BEPS:  [GitHub Repository](https://github.com/yongtaoliu/AE-PostExperimentAnalysis-DKL-BEPS)


## Cite this work as
```
@Article{D5DD00033E,
author ="Pratiush, Utkarsh and Roccapriore, Kevin M. and Liu, Yongtao and Duscher, Gerd and Ziatdinov, Maxim and Kalinin, Sergei V.",
title  ="Building workflows for an interactive human-in-the-loop automated experiment (hAE) in STEM-EELS",
journal  ="Digital Discovery",
year  ="2025",
pages  ="-",
publisher  ="RSC",
doi  ="10.1039/D5DD00033E",
url  ="http://dx.doi.org/10.1039/D5DD00033E",
abstract  ="Exploring the structural{,} chemical{,} and physical properties of matter on the nano- and atomic scales has become possible with the recent advances in aberration-corrected electron energy-loss spectroscopy (EELS) in scanning transmission electron microscopy (STEM). However{,} the current paradigm of STEM-EELS relies on the classical rectangular grid sampling{,} in which all surface regions are assumed to be of equal a priori interest. However{,} this is typically not the case for real-world scenarios{,} where phenomena of interest are concentrated in a small number of spatial locations{,} such as interfaces{,} structural and topological defects{,} and multi-phase inclusions. One of the foundational problems is the discovery of nanometer- or atomic-scale structures having specific signatures in EELS spectra. Herein{,} we systematically explore the hyperparameters controlling deep kernel learning (DKL) discovery workflows for STEM-EELS and identify the role of the local structural descriptors and acquisition functions in experiment progression. In agreement with the actual experiment{,} we observe that for certain parameter combinations the experiment path can be trapped in the local minima. We demonstrate the approaches for monitoring the automated experiment in the real and feature space of the system and knowledge acquisition of the DKL model. Based on these{,} we construct intervention strategies defining the human-in-the-loop automated experiment (hAE). This approach can be further extended to other techniques including 4D STEM and other forms of spectroscopic imaging. The hAE library is available on Github at https://github.com/utkarshp1161/hAE/tree/main/hAE."}

```
