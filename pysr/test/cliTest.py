import io
import subprocess
import sys
import unittest


def run_command(command):
    return subprocess.run(command.split(" "), stdout=subprocess.PIPE).stdout.decode("utf-8").replace("\r\n", "\n")


def make_command(command):
    return "\n".join(command)


class TestCli(unittest.TestCase):

    def test_help_on_all_commands(self):
        command_to_test = "python -m pysr --help"
        expected_lines = ["Usage: pysr [OPTIONS] COMMAND [ARGS]...",
                          "",
                          "Options:",
                          "  --help  Show this message and exit.",
                          "",
                          "Commands:",
                          "  install  Install Julia dependencies for PySR.",
                          ""]

        expected = make_command(expected_lines)
        actual = run_command(command_to_test)
        self.assertEqual(expected, actual)  # add assertion here

    def test_help_on_install(self):
        command_to_test = "python -m pysr install --help"
        expected_lines = ["Usage: pysr install [OPTIONS]",
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
                          ""]

        expected = make_command(expected_lines)
        actual = run_command(command_to_test)
        self.assertEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()
