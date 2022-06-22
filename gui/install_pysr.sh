import os

# Install Julia:
if [ ! -d "/home/user/julia" ]; then
    wget https://julialang-s3.julialang.org/bin/linux/x64/1.7/julia-1.7.3-linux-x86_64.tar.gz
    tar zxvf julia-1.7.3-linux-x86_64.tar.gz
    mkdir /home/user/julia
    mv julia-1.7.3/* /home/user/.local/
fi

# Need to install PySR in separate python instance:
if [ ! -d "/home/user/.julia/environments/pysr-0.9.3" ]; then
    export PATH="$PATH:/home/user/julia/bin/"
    python -c 'import pysr; pysr.install()'
fi