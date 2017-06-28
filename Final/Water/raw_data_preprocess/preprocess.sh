#!/usr/bin/env bash

echo "place the raw files and the two PreProcess .py files in the same directory with this one.\n"
echo "This will generate test.py and train_data.py for the next step."

python3 PreProcess_train_label.py
python3 PreProcess_train_value_and_test.py