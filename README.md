# Hello!

This is a simple repository with example scripts, code and baselines to train and evaluate models on AddBiomechanics data.

## Getting The Data

1. First you need to install the AddBiomechanics Command Line Interface (CLI) `addb`. There are instructions [here](https://github.com/keenon/AddBiomechanics/tree/main/cli).
2. Then, you need to download the data. You can do this by running `./update_dataset.sh` from the root directory of this repository. It will ask for your username and password to log in to AddBiomechanics, if you haven't used the `addb` tool before, and then some confirmations.

Once you've completed these steps, there will be a dataset in `data/processed/`, using a standard armless Rajagopal skeleton format. The data is contained in `*.bin` files, which can be read with the `nimblephysics.biomechanics.SubjectOnDisk` class.

## Running the Baseline

First, run `pip3 install -r requirements.txt`

Then to run a few epochs of training:

`python3 main.py`

That will print out results as it trains.