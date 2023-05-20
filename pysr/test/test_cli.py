import subprocess
import unittest


def run_command(command):
    """
    Retrieve output of a command string, decode and convert from CRLF to LF formatting
    """
    return (
        subprocess.run(command.split(" "), stdout=subprocess.PIPE)
        .stdout.decode("utf-8")
        .replace("\r\n", "\n")
    )


class TestCli(unittest.TestCase):
    def test_help_on_all_commands(self):
        command_to_test = "python -m pysr --help"
        expected_lines = [
            "Usage: pysr [OPTIONS] COMMAND [ARGS]...",
            "",
            "Options:",
            "  --help  Show this message and exit.",
            "",
            "Commands:",
            "  install  Install Julia dependencies for PySR.",
            "",
        ]

        expected = "\n".join(expected_lines)
        actual = run_command(command_to_test)
        self.assertEqual(expected, actual)

    def test_help_on_install(self):
        command_to_test = "python -m pysr install --help"
        expected_lines = [
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

        expected = "\n".join(expected_lines)
        actual = run_command(command_to_test)
        self.assertEqual(expected, actual)


def runtests():
    """Run all tests in cliTest.py."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestCli))
    runner = unittest.TextTestRunner()
    return runner.run(suite)
