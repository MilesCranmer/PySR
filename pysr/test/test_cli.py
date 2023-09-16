import unittest

from click import testing as click_testing

from .._cli.main import pysr


class TestCli(unittest.TestCase):
    # TODO: Include test for custom project here.
    def setUp(self):
        self.cli_runner = click_testing.CliRunner()

    def test_help_on_all_commands(self):
        expected = "\n".join(
            [
                "Usage: pysr [OPTIONS] COMMAND [ARGS]...",
                "",
                "Options:",
                "  --help  Show this message and exit.",
                "",
                "Commands:",
                "  install  Install Julia dependencies for PySR.",
                "",
            ]
        )
        result = self.cli_runner.invoke(pysr, ["--help"])
        self.assertEqual(expected, result.output)
        self.assertEqual(0, result.exit_code)

    def test_help_on_install(self):
        expected = "\n".join(
            [
                "Usage: pysr install [OPTIONS]",
                "",
                "  Install Julia dependencies for PySR.",
                "",
                "Options:",
                "  -p, --project PROJECT_DIRECTORY",
                "                                  Install in a specific Julia project (e.g., a",
                "                                  local copy of SymbolicRegression.jl).",
                "  -q, --quiet                     Disable logging.",
                "  --precompile                    Force precompilation of Julia libraries.",
                "  --no-precompile                 Disable precompilation.",
                "  --help                          Show this message and exit.",
                "",
            ]
        )
        result = self.cli_runner.invoke(pysr, ["install", "--help"])
        self.assertEqual(expected, result.output)
        self.assertEqual(0, result.exit_code)


def runtests():
    """Run all tests in cliTest.py."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestCli))
    runner = unittest.TextTestRunner()
    return runner.run(suite)
