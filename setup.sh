#!/usr/bin/env bash

if [ ! -d data/glove.6B ]; then
    mkdir -p data/glove.6B
    wget -P data/glove.6B http://nlp.stanford.edu/data/glove.6B.zip
    unzip data/glove.6B/glove.6B.zip -d data/glove.6B
    rm data/glove.6B/glove.6B.zip
fi

if [ ! -d data/snli_1.0 ]; then
    wget -P data https://nlp.stanford.edu/projects/snli/snli_1.0.zip
    unzip data/snli_1.0.zip -d data
    rm data/snli_1.0.zip
    rm -rf data/__MACOSX
fi
