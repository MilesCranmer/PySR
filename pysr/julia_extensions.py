"""This file installs and loads extensions for SymbolicRegression."""
from .julia_import import jl


def load_required_packages(*, turbo=False, enable_autodiff=False):
    if turbo:
        load_package("LoopVectorization")
    if enable_autodiff:
        load_package("Zygote")
    if cluster_manager is not None:
        load_package("ClusterManagers")


def load_package(package_name):
    jl.seval(f"""
        try
            using {package_name}
        catch e
            isa(e, ArgumentError) || throw(e)
            using Pkg: Pkg
            Pkg.add("{package_name}")
            using {package_name}
        end
    """)
    return None
