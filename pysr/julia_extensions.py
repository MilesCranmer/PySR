"""This file installs and loads extensions for SymbolicRegression."""

from typing import Optional

from .julia_helpers import jl_array
from .julia_import import Pkg, jl

UUIDs = {
    "LoopVectorization": "bdcacae8-1622-11e9-2a5c-532679323890",
    "Bumper": "8ce10254-0962-460f-a3d8-1f77fea1446e",
    "Zygote": "e88e6eb3-aa80-5325-afca-941959d7151f",
    "MPIClusterManagers": "e7922434-ae4b-11e9-05c5-9780451d2c66",
    "ClusterManagers": "34f1f09b-3a8b-5176-ab39-66d58a4d544e",
}


def load_required_packages(
    *,
    turbo: bool = False,
    bumper: bool = False,
    enable_autodiff: bool = False,
    cluster_manager: Optional[str] = None,
):
    if turbo:
        load_package("LoopVectorization")
    if bumper:
        load_package("Bumper")
    if enable_autodiff:
        load_package("Zygote")
    if cluster_manager is not None:
        if cluster_manager == "mpi":
            load_package("MPIClusterManagers")
        else:
            load_package("ClusterManagers")


def load_all_packages():
    """Install and load all Julia extensions available to PySR."""
    specs = [Pkg.PackageSpec(name=key, uuid=value) for key, value in UUIDs.items()]
    Pkg.add(jl_array(specs))
    Pkg.resolve()
    jl.seval("import " + ", ".join(UUIDs.keys()))


# TODO: Refactor this file so we can install all packages at once using `juliapkg`,
#       ideally parameterizable via the regular Python extras API


def isinstalled(package_name: str):
    return jl.haskey(Pkg.dependencies(), jl.Base.UUID(UUIDs[package_name]))


def load_package(package_name: str) -> None:
    if not isinstalled(package_name):
        Pkg.add(name=package_name, uuid=UUIDs[package_name])
        Pkg.resolve()

    # TODO: Protect against loading the same symbol from two packages,
    #       maybe with a @gensym here.
    jl.seval(f"import {package_name}")
    return None
