# Welcome to PySR's contributing guide <!-- Forked from GitHub's awesome template contributing guide -->

Thank you for investing your time in contributing to our project! Any contribution you make will be reflected on the contributors lists: [frontend](https://github.com/MilesCranmer/PySR/graphs/contributors) and [backend](https://github.com/MilesCranmer/SymbolicRegression.jl/graphs/contributors) :sparkles:.

In this guide you will get an overview of the contribution workflow from opening an issue, creating a PR, reviewing, and merging the PR.

## New contributor guide

To get an overview of the project, read PySR's [README](README.md). The [PySR docs](https://astroautomata.com/PySR/) give additional information.
Here are some resources to help you get started with open source contributions in general:

- [Finding ways to contribute to open source on GitHub](https://docs.github.com/en/get-started/exploring-projects-on-github/finding-ways-to-contribute-to-open-source-on-github)
- [Set up Git](https://docs.github.com/en/get-started/quickstart/set-up-git)
- [GitHub flow](https://docs.github.com/en/get-started/quickstart/github-flow)
- [Collaborating with pull requests](https://docs.github.com/en/github/collaborating-with-pull-requests)

### Issues

#### Create a new issue

If you spot a problem with PySR, [search if an issue already exists](https://docs.github.com/en/github/searching-for-information-on-github/searching-on-github/searching-issues-and-pull-requests#search-by-the-title-body-or-comments). If a related issue doesn't exist, you can open a new issue using a relevant [issue form](https://github.com/MilesCranmer/PySR/issues/new/choose).

#### Solve an issue

Scan through our [existing issues](https://github.com/MilesCranmer/PySR/issues) to find one that interests you (feel free to work on any!). You can narrow down the search using `labels` as filters. See [Labels](/contributing/how-to-use-labels.md) for more information. If you find an issue to work on, you are welcome to open a PR with a fix.

### Make Changes

#### Make changes locally

1. Fork the repository.
- Using GitHub Desktop:
  - [Getting started with GitHub Desktop](https://docs.github.com/en/desktop/installing-and-configuring-github-desktop/getting-started-with-github-desktop) will guide you through setting up Desktop.
  - Once Desktop is set up, you can use it to [fork the repo](https://docs.github.com/en/desktop/contributing-and-collaborating-using-github-desktop/cloning-and-forking-repositories-from-github-desktop)!

- Using the command line:
  - [Fork the repo](https://docs.github.com/en/github/getting-started-with-github/fork-a-repo#fork-an-example-repository) so that you can make your changes without affecting the original project until you're ready to merge them.

2. Create a working branch and start with your changes!

3. (Optional) If you would like to make changes to PySR itself, skip to step 4. However, if you are interested in making changes to the _symbolic regression code_ itself,
check out the [guide](https://astroautomata.com/PySR/backend/) on modifying a custom SymbolicRegression.jl library.
In this case, you might instead be interested in making suggestions to the [SymbolicRegression.jl](http://github.com/MilesCranmer/SymbolicRegression.jl) library.

4. You can install your local version of PySR with `python setup.py install`, and run tests with `python -m pysr.test main`.

### Commit your update

Once you are happy with your changes, run `black .` to apply [Black](https://black.readthedocs.io/en/stable/) formatting to your local
version. Commit the changes once you are ready.

### Pull Request

When you're finished with the changes, create a pull request, also known as a PR.
- Don't forget to [link PR to issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue) if you are solving one.
- Enable the checkbox to [allow maintainer edits](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/allowing-changes-to-a-pull-request-branch-created-from-a-fork) so the branch can be updated for a merge.
Once you submit your PR, a PySR team member will review your proposal. We may ask questions or request additional information.
- We may ask for changes to be made before a PR can be merged, either using [suggested changes](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/incorporating-feedback-in-your-pull-request) or pull request comments. You can apply suggested changes directly through the UI. You can make any other changes in your fork, then commit them to your branch.
- As you update your PR and apply changes, mark each conversation as [resolved](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/commenting-on-a-pull-request#resolving-conversations).
- If you run into any merge issues, checkout this [git tutorial](https://github.com/skills/resolve-merge-conflicts) to help you resolve merge conflicts and other issues.

### Your PR is merged!

Congratulations :tada::tada: The PySR team thanks you :sparkles:.

Once your PR is merged, your contributions will be publicly visible.

Thanks for being part of the PySR community!
