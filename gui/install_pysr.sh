import os

# Install Julia:
if [ ! -f "/home/user/.local/bin/julia" ]; then
    wget https://raw.githubusercontent.com/abelsiqueira/jill/main/jill.sh
    chmod a+x jill.sh
    ./jill.sh --version 1.7.3 -y
fi

# Need to install PySR in separate python instance:
if [ ! -d "/home/user/.julia/environments/pysr-0.9.3" ]; then
    export PATH="$PATH:/home/user/julia/bin/"
    python -c 'import pysr; pysr.install()'
fi