# SynSemNet

This repository defines deep neural models for training and generating contextualized word representations that
factor ("disentangle") syntactic and semantic information using multi-task adversarial learning. We currently use
four tasks, two designed to favor semantic information and two designed to favor syntactic information:

- Syntactic tasks:
  - Constituent parsing (supervised)
  - Word position loss (unsupervised)
  
- Semantic tasks:
  - Semantic text similarity
  - Bag-of-words loss (unsupervised)

The supervised tasks are intended to require rich representations available from human annotators, while the
unsupervised tasks (by which we mean that they only require unannotated text) are intended to be general enough to
moderate any systematic biases imposed by the supervised tasks.

We train two contextualized word encoders, one syntactic and one semantic. Each encoder is optimized to generate
word encodings that support the target representation type and that are orthogonal to the non-target representation
type. This is achieved through domain-adversarial training (Ganin et al., 2015): the syntactic encoder gets ordinary
gradients from the parsing and word position tasks and domain-adversarial gradients from the semantic and
bag-of-words tasks, while the semantic encoder gets ordinary gradients from the semantic and bag-of-words tasks and 
domain adversarial gradients from the parsing and word position tasks. The goal is to obtain semantic encodings with
little information about syntax and syntactic encodings with little information about semantics.

## Installation

This repository requires Tensorflow (and all Tensorflow prereqs). All development work has been done using Python 3.
Python 2 compatibility is possible but untested.

After cloning, navigate to the SynSemNet root and run

    git clone git@github.com:coryshain/tree2labels.git;
    cd tree2labels/EVALB;
    make all

This sets up code needed to generate and evaluate the parse labels (it's a fork of the code from 
Gómez-Rodríguez and Vilares, 2018, to enable Python 3 support).

## Data Setup

We train on the WSJ portion of [Penn Treebank 3](https://catalog.ldc.upenn.edu/LDC99T42) for parsing and 
[STSbenchmark](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark) for STS. The unsupervised tasks simply use the
sentences from these datasets for training. We have provided a setup utility for PTB preprocessing. For usage
 details, run:

    python3 -m synsemnet.datasets.wsj.build -h

## Fitting Models

Model data and hyperparameters are defined in `*.ini` config files. For an example config file, see `ssn_model.ini`
in the repository root. For a full description of all settings that can be controlled with the config file,
see the SynSemNet initialization params by running:

    python3 -m synsemnet.bin.help
    
Once you have defined an `*.ini` config file, fit the model by running the following from the repository root:

    python3 -m synsemnet.bin.train <PATH>.ini

## Terminology

Throughout this codebase, we assume the following definitions in variable names

- `embedding`: A context-independent dense representation of data (words or characters)
- `encoding`: A context-dependent representation of data
- `encoder`: A neural transform that generates an encoding of data
- `decoder`: A neural transform that generates a sequential representation from a fixed-dimensional encoding
- `classifier`: A neural transform that generates a non-sequential representation from a fixed-dimensional encoding
- `syn`/`syntactic`: Having to do with the syntactic encoder (regardless of task)
- `sem`/`semantic`: Having to do with the semantic encoder (regardless of task)
