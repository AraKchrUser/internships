#!/bin/bash

pip install -r requirements.txt > /dev/null 2>&1
python phase_vocoder.py $1 $2 $3