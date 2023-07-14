#!/bin/bash
set -e

mkdir -p data
pushd data
mkdir -p raw

pushd raw
echo "Download the data from the DEV servers that has been standardized to the rajagopal_no_arms.osim skeleton:"
addb -d dev download "standardized/rajagopal_no_arms/.*\.bin$" --marker-error-cutoff 0.035
echo "Download the data from the PROD servers that has been standardized to the rajagopal_no_arms.osim skeleton:"
addb -d prod download "standardized/rajagopal_no_arms/.*\.bin$" --marker-error-cutoff 0.035
popd

echo "Post-process the data to low-pass filter at 20 Hz and standardize at 100 Hz:"
addb post-process raw processed --lowpass-hz 20 --sample-rate 100 --trim-to-grf True
popd

rm -rf data/train
rm -rf data/dev
python3 create_splits.py