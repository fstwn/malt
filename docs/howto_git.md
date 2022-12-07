# Getting started with git

Git is a free and open source distributed version control system. It is at the
heart of GitHub so to speak (hence the name). You need an installation of git
to collaborate on repositories or simply update your local repositories in
an efficient and easy way. There are two possibilities to use git in the
context of working with this repository:

## 1. The GitHub desktop client

If you want to interact with the repository in a user-fiendly way without
using the command line, you can use [GitHub Desktop](https://desktop.github.com/).
**Please note that using git via the command line (i.e. `git pull`) will not
work out of the box when using GitHub Desktop. You'll have to use the graphical
interface of the client instead!**

## 2. The Git command line

If you feel comfortable working with the command line, you can use the original
distrbution of [Git](https://git-scm.com/). If you want to use the Windows
command line or Powershell with git, make sure to enable this option during
the installation process!

You can find a more in-depth guide on how to get started with git [here](https://github.com/git-guides/install-git).

There is also a cheat-sheet for working with git available [here](https://about.gitlab.com/images/press/git-cheat-sheet.pdf).

## 3. Collaboration using Git

The main purposes of git are version control and, of course, collaboration
with multiple people on one codebase. This can get very complex, so we rely
on git to make the process manageable.

To get started with your own contributions to a repository, you have to
get familiar with *branches*. Branches allow us to store multiple versions
of the codebase separately from one another, so every collaborator can work
on their version of the codebase without messing up the production code or the
code of others.

At first, every repository has only one branch. In our case this branch is
called `main`[^1], because it is (you might have guessed it) the main branch.
The main branch is supposed to store the current, working version of the
codebase.

If you want to contribute to this repository in a meaningful way, you have to
create your own development branch. When you have developed some new features
that are ready, you can open a `pull request` to merge them from your
development branch into the main branch.

### 3.1 Creating a development branch for your code contribution

Okay, say you have written some code and want to contribute it to the
repository - great! Let's get started:

**IMPORTANT:** *Before continuing, please backup your code to make sure it
won't get lost during the process of creating and switching to you development
branch*

To create a new branch for your developments, run the following command
while having the current working directory set to the directory of the
`malt` repository:

```
git branch dev_yourname
```

This will create a branch with the name `dev_yourname`. Of course please feel
free to name this branch however you like. Next, we need to switch to that
branch to start working on it:

```
git checkout dev_yourname
```

This will switch the current working branch to `dev_yourname`. Note that if you
have uncommitted changes before switching branches, this action will not work!






[^1]: A little bit of history: Formerly, the main branch was commonly referred
to as *master* branch. The *master* and *slave* metaphors in technology and
especially in software development date back to the early 20th century. In
recent years this language, which blatantly reproduces racism, has been and is
being questioned (and rightfully so!). So, I will not use this terminology
anymore. Better words to use are for example *main*, *primary/replica* or
*leader/follower*.