#!/bin/bash
sudo python3 -m pip install numpy pandas &&
    sudo python3 setup.py install &&
    python3 test/test.py

