# Getting started with Anaconda

Conda is an open-source package and environment management system that runs on
Windows, macOS, and Linux. Conda quickly installs, runs, and updates packages
and their dependencies. It also easily creates, saves, loads, and switches
between environments on your local computer. It was created for Python
programs, but it can package and distribute software for any language.

## 1. Installing Anaconda

The first step is downloading and installing Anaconda. You can download the
necessary installer [here](https://www.anaconda.com/products/distribution).
During installation, I recommend **adding conda to your PATH environment
variable**.

## 2. Setting Anaconda up to work with PowerShell

Conda comes with its own command prompt, called the *Anaconda Prompt*. In my
opinion, it is much more convenient to have conda available in the Windows
Powershell (but this might be a matter of taste). If you do *NOT* set up
Anaconda to work with the Powershell, you will have to use the Anaconda Prompt
for working with Anaconda (i.e. for activating a virtual environment with 
`conda activate`).

If you want to use Anaconda with the Powershell, you can do it like this:

1. Open a `Anaconda Powershell Prompt` from the Windows Start Menu.

2. Now run `conda init powershell`.

3. Close the Anaconda Powershell and open a *regular* Powershell.

4. You *might* get an error like this:

```
\WindowsPowerShell\profile.ps1 cannot be loaded because running scripts is disabled on this system. For more information, see about_Execution_Policies at https:/go.microsoft.com/fwlink/?LinkID=135170. At line:1 char:3
```

5. If not, you`re all set. If you get the error, run this in Powershell:

```
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned
```

