---
name: Bug report
about: Create a report to help us improve
title: "[BUG] *Summary of bug*"
labels: bug
assignees: MilesCranmer

---

**Describe the bug**
A clear and concise description of what the bug is.

**Version (please include the following information):**
- OS: [e.g. macOS]
- Julia version [Run `julia --version` in the terminal]
- Python version [Run `python --version` in the terminal]
- Did you install with `pip` or `conda`?
- PySR version [Run `python -c 'import pysr; print(pysr.__version__)'`]
- Does the bug still appear with the latest version of PySR?

**Configuration**
- What are your PySR settings?
- What dataset are you running on?
- If possible, please share a minimal code example that produces the error.

**Error message**
Add the error message here, or whatever other information would be useful for debugging.

If the error is "Couldn't find equation file...", this error indicates something
went wrong with the backend. Please scroll up and copy
the output of Julia, rather than the output of python.

**Additional context**
Add any other context about the problem here.
