import click
from ..julia_helpers import install


@click.group("cli")
@click.pass_context
def cli(context):
    ctx = context


@cli.command("install", help="Install Julia dependencies for PySR")
@click.option("-p", "--project", default=None, type=str)
@click.option("-q", "--quiet", is_flag=True, default=False)
@click.option("--precompile", 'precompile', flag_value=True, default=None)
@click.option("--no-precompile", 'precompile', flag_value=False, default=None)
def _install(julia_project, quiet, precompile):
    install(julia_project, quiet, precompile)
