"""This file installs and loads extensions for SymbolicRegression."""

from typing import Optional

from .julia_import import Pkg, jl


def load_required_packages(
    *,
    turbo: bool = False,
    bumper: bool = False,
    enable_autodiff: bool = False,
    cluster_manager: Optional[str] = None,
):
    if turbo:
        load_package("LoopVectorization", "bdcacae8-1622-11e9-2a5c-532679323890")
    if bumper:
        load_package("Bumper", "8ce10254-0962-460f-a3d8-1f77fea1446e")
    if enable_autodiff:
        load_package("Zygote", "e88e6eb3-aa80-5325-afca-941959d7151f")
    if cluster_manager is not None:
        load_package("ClusterManagers", "34f1f09b-3a8b-5176-ab39-66d58a4d544e")


def isinstalled(uuid_s: str):
    return jl.haskey(Pkg.dependencies(), jl.Base.UUID(uuid_s))


def load_package(package_name: str, uuid_s: str) -> None:
    if not isinstalled(uuid_s):
        Pkg.add(name=package_name, uuid=uuid_s)

    # TODO: Protect against loading the same symbol from two packages,
    #       maybe with a @gensym here.
    jl.seval(f"using {package_name}: {package_name}")
    return None
