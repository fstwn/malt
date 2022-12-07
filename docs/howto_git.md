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

**FIRST OFF:** *When working with repositories and git, especially when more
people come in, I strongly suggest using the source control function of a
code editor like VSCode. It will be a lot easier and less intimidating than
doing everything over the commandline **BUT** you should already be familiar
with the basic actions that git provides, so I recommend reading this until
the end before doing anything.*

To get started with your own contributions to a repository, you have to
get familiar with *branches*. Branches allow us to store multiple versions
of the codebase separately from one another, so every collaborator can work
on their version of the codebase without messing up the production code or the
code of other collaborators.

At first, every repository has only one branch. In our case this branch is
called `main`[^1], because it is (you might have guessed it) the main branch.
The main branch is supposed to store the current, working version of the
codebase.

If you want to contribute to this repository in a meaningful way, you have to
create your own development branch. When you have developed some new features
that are ready, you can use a `pull request` to merge the changes from your
development branch into the main branch.

### 3.1 Creating a development branch for your code contribution

Okay, say you have written some code and want to contribute it to the
repository - great! Let's get started:

**IMPORTANT:** *Before continuing, please backup your code to make sure it
won't get lost during the process of creating and switching to your development
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

### 3.2 Staging changes and commiting to your development branch

Since git is a version control system, it will track all changes made to files
within the repository. That means if you just have written some new code, you
need to 'save' these changes so that they are indexed within the repository.
In git, this operation is called `commit`.

A commit usually consists of some changes to files, new files and/or deleted
files. Let's make an example:

- You have created a development branch called `dev_yourname` for your
contributions and switched to it.
- You have written some new code in a new file `myscript.py`, which is inside
the `src` directory.
- You now want to commit these changes to your development branch to 'save'
the current state of things.

The first step is *staging* your changes for the commit. The `git add` command
adds a change in the working directory to the staging area. It tells Git that
you want to include updates to a particular file in the next commit. However,
`git add` doesn't really affect the repository in any significant way —
changes are not actually recorded until you run `git commit`.
You can do this by running:

```
git add src/myscript.py
```

Now, your changes to this file are staged for the next commit. Of course you
can still add more changes to the staging area, for example if you have made
changes to another file.

It is also possible to stage *all* current changes at once. To do this, you
can run:

```
git add --all
```

Once you have staged some changes, it's time for your first commit! The commit
is used for telling Git *"Yes, I have made those changes, I want to save this
state of things"*. A commit should *always* include a commmit message. Think of
it as keeping a tiny diary about the things that you have done to the code.
You can do your first commit just like this:

```
git commit -m "Added a new fancy script doing some fancy stuff - yay!"
```

**Hooooray!** You have just made the first commit to you *local* version of the
repository! You have successfully learnt about branches, how to stage changes
*and* how to commit!

### 3.3 Pushing changes to the remote

Now there is only one left thing to do: Pushing your changes to the remote
repository (e.g. on GitHub). But first, a word of caution...

***ATTENTION:*** **Pushing changes is a potentially dangerous operation. Thus,
if your name is not Max Benjamin Eschenbach, please *NEVER, EVER EVER EVER*
push anything to the `main` branch! Follow the coming section on pull requests
instead to learn how to safely contribute to the `main` branch. Thank you :-)**

So, once you have commited your changes to your newly created local development
branch, we are going to push these changes *and* the branch to the remote. If
you are pushing **the first commit on your development branch** run:

```
git push -u origin dev_yourname
```

This command will push your local development branch with your changes to the
remote (i.e. on GitHub). The `-u` flag tells git to set that remote branch as
*upstream* for your local branch, meaning that your local branch and the remote
branch are 'linked' now. You only have to use `-u` when first pushing a new
branch. After that, you can just push like this:

```
git push origin dev_yourname
```

Awesome! Your development branch and your changes are now also saved within the
remote repository. Thank you for your contribution!

### 3.4 Pull requests

Once you have done all the above steps, you can open a pull request. This is
like saying *"Hey, I have written some cool stuff in my dev branch, can you
please merge it to the main branch?"*

Personally, I recommend doing pull requests using the web interface of GitHub.
For this reason, I will simply link the official GitHub tutorial on that:

[Creating a pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request)

If you have any questions about pull requests, of course feel free to ask.

### 3.5 Last but not least

As said in the beginning, I *strongly* recommend managing branches, staging and
commits using a graphical interface. Personally, I recommend using the source
control feature of VSCode. Once you arrived at this section, you should know
enough about git to work with it in a pretty straightforward way, so I'm
simply linking the tutorial:

[Using Git source control in VS Code](https://code.visualstudio.com/docs/sourcecontrol/overview)


[^1]: A little bit of history: Formerly, the main branch was commonly referred
to as *master* branch. The *master* and *slave* metaphors in technology and
especially in software development date back to the early 20th century. In
recent years this language, which blatantly reproduces racism, has been and is
being questioned (and rightfully so!). So, I will not use this terminology
anymore. Better words to use are for example *main*, *primary/replica* or
*leader/follower*.

There is also a cheat-sheet for working with git available [here](https://about.gitlab.com/images/press/git-cheat-sheet.pdf).
