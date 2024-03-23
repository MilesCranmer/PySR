"""This file installs and loads extensions for SymbolicRegression."""

from .julia_import import jl


def load_required_packages(
    *, turbo=False, bumper=False, enable_autodiff=False, cluster_manager=None
):
    if turbo:
        load_package("LoopVectorization", "bdcacae8-1622-11e9-2a5c-532679323890")
    if bumper:
        load_package("Bumper", "8ce10254-0962-460f-a3d8-1f77fea1446e")
    if enable_autodiff:
        load_package("Zygote", "e88e6eb3-aa80-5325-afca-941959d7151f")
    if cluster_manager is not None:
        load_package("ClusterManagers", "34f1f09b-3a8b-5176-ab39-66d58a4d544e")


def load_package(package_name, uuid):
    jl.seval(
        f"""
        try
            using {package_name}
        catch e
            isa(e, ArgumentError) || throw(e)
            using Pkg: Pkg
            Pkg.add(name="{package_name}", uuid="{uuid}")
            using {package_name}
        end
    """
    )
    return None
