import os

# Install Julia:
if [ ! -f "/home/user/.local/bin/julia" ]; then
    bash -ci "$(curl -fsSL https://raw.githubusercontent.com/abelsiqueira/jill/main/jill.sh)"
fi

# Need to install PySR in separate python instance:
if [ ! -d "/home/user/.julia/environments/pysr-0.9.3" ]; then
    export PATH="$PATH:/home/user/julia/bin/"
    python -c 'import pysr; pysr.install()'
fi