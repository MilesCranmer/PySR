# This builds a dockerfile containing a working copy of PySR
# with all pre-requisites installed.

FROM julia:1.8.2

RUN julia -e 'using Pkg; Pkg.add("Conda"); Pkg.build("Conda")'
