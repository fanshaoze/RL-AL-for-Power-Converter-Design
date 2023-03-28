# RL-AL-for-Power-Converter-Design

## Table of Contents

- [Description](#Description)
- [Environment](#environment)
- [Representation](#representation)
- [Structure](#structure)
    - [Framework](#framework)
    - [File Structure](#file-structure)
- [Surrogate Model](#surrogate-model)
	- [Setting](#setting)
	- [Example: Model Training](#Example-Model-Training)
- [RL-based Converter Design](#rl-based-converter-design)
	- [Setting](#setting)
	- [Example: Converter Design](#Example-Converter-Design)
- [Results](#results)


## Description
This is the program of UCT, parallel UCT and Genetic search.

## Environment

Operating system: Linux  
Python: Python3.6, Python 3.7 or Python 3.8  
Package: ngspice, networkx, numpy and matplotlib
```sh
$ sudo apt install ngspice
$ pip3 install networkx numpy matplotlib
```
## Representation

**max_episode_length**: The max allowed step for simulator to take action  
**deterministic**: True if when a state take an action, the next state is deterministic. 
False if the next state can be one of several different states  
**ucb_scalar**: the ucb scalar, currently it should be set in range 5 to 20   
**gamma**: The parameter we multiplied when we do the back propagation in UCT.
In this application we set it as 1.  

## Structure

### Framework


### File Structure

## Surrogate Model

### Setting

### Example: Model Training


## RL-based Converter Design

### Setting

### Example: Converter Design
