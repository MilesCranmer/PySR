#!/bin/bash
julia -e 'import Pkg; Pkg.add("Optim"); Pkg.add("SpecialFunctions")'
sudo python3 -m pip install numpy pandas
sudo python3 setup.py install
python3 test/test.py

