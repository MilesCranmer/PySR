# This builds a dockerfile containing a working copy of PySR
# with all pre-requisites installed.


ARG VERSION=latest
FROM julia:$VERSION

RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    build-essential python3 python3-dev python3-pip python3-setuptools \
    vim git wget curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /pysr

# Caches install (https://stackoverflow.com/questions/25305788/how-to-avoid-reinstalling-packages-when-building-docker-image-for-python-project)
ADD ./requirements.txt /pysr/requirements.txt
RUN pip3 install -r /pysr/requirements.txt

# Install PySR:
ADD . /pysr/
RUN pip3 install .

# Install Julia pre-requisites:
RUN julia -e 'using Pkg; Pkg.add("SymbolicRegression")'

# Install IPython and other useful libraries:
RUN pip3 install ipython jupyter matplotlib

CMD ["ipython"]