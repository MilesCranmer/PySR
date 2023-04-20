import click
from ..julia_helpers import install


@click.group("cli")
@click.pass_context
def cli(context):
    ctx = context


@cli.command("install", help="Install required Julia dependencies")
@click.option("--julia_project", default=None)
@click.option("--quiet", default=False)
@click.option("--precompile", default=None)
def _install(julia_project, quiet, precompile):
    install(julia_project, quiet, precompile)
