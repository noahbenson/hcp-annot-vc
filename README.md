# Annotation of Visual Cortices of the Human Connectome Project

This repository is a tool for annotating the iso-eccentricity and visual area boundary contours in the visual cortices of the subjects from the Human Connectome Project 7 Tesla Retinotopy Dataset ([Benson et al., 2018](https://doi.org/10.1167/18.13.23); see also:
[Benson **et al**., 2021](https://doi.org/10.1101/2020.12.30.424856)).

## How to use this repository

### Before you start

Using this repository requires signing up for GitHub and downloading/running some software, but all of these are free.

1. You must have a GitHub account ([join here](https://github.com/join)) and you must have `git` installed ([instructions here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)).
2. These instructions are written for use with a BASH shell for entering commands. You may have to change these instructions slightly if you are using a different shell, or you can typically start a BASH shell by typing `bash` in another shell. [This page](https://swcarpentry.github.io/shell-novice/) contains a good introduction to the Shell, and the "Setup" link explains how to install it and get it working (you can skip the zip-file download at the top of the instructions and just follow the OS-specific instructions at the bottom).
3. You must have Docker installed ([instructions here](https://docs.docker.com/get-docker/)). Docker is a virtual-machine tool that allows you to execute code packaged up in docker images by other authors without having to share an OS or software dependencies.

### Instructions

#### Setup (only follow these instructions once)

1. Make a fork of this repository into your own GitHub profile. This is done using the "Fork" button in the upper-right corner of this repository's GitHub page.
2. In a BASH shell, clone your own version of this repository.

   ```bash
   $ git clone https://github.com/<username>/hcp-annot-vc
   ```
   
   (Be sure to replace the "`<username>`" in the previous command with your GitHub username.)
   
   Then `cd` into the new directory.
   
   ```bash
   $ cd hcp-annot-vc
   ```

#### Running the Tool (follow these each time you work on the project)

3. Update the repository with any new changes. To do this, start by going to **your** copy of the repository: `http://github.com/<username>/hcp-annot-vc` where `<username>` is your GitHub username. There should be a green button named "Code" in the upper-right section of this page. Just below this button, there will be a button labeled "Fetch Upstream" with a little arrow next to it. Click the arrow and select "Fetch and Merge" to update your repository. If you are asked to enter a merge message, you can safely leave this message unchanged. If you run this "Fetch and Merge" step, you then need to run the following in the command line:
   
   ```bash
   $ git pull
   ```
   
   If there was no "Fetch Upstream" button, you can skip this step.
   
4. Run `docker compose up --build` to start the virtual-machine.

   ```bash
   $ docker compose up --build
   ```
   
   This command will likely take a long time to run the first time you run it, and it will produce a lot of output. Near the end of the output will be something that looks like this:
   
   ```
   neuropythy_1  |     To access the notebook, open this file in a browser:
   neuropythy_1  |         file:///home/jovyan/.local/share/jupyter/runtime/nbserver-9-open.html
   neuropythy_1  |     Or copy and paste one of these URLs:
   neuropythy_1  |         http://ba0d0b5346d7:8888/?token=3e27b357b6a63752d130ae22ac0c160d8e26ec252c4d3c66
   neuropythy_1  |      or http://127.0.0.1:8888/?token=3e27b357b6a63752d130ae22ac0c160d8e26ec252c4d3c66
   ```
   
   This indicates that the virtual machine is running.
   
   If for some reason you think that your docker instance is not building the annotation tool correctly, or that something about the build is not being updated, you can run the following command before the `docker compose up --build` command to ensure that the docker image is rebuilt cleanly: `docker build --no-cache .`.
5. Copy-and-paste the web address starting with `http://127.0.0.1:8888/` into your web-browser. This should bring up a [Jupyter](https://jupyter.org/) interface.
6. Click to open the file `open_me.ipynb`. This should open a new window containing a Jupyter notebook.
6. Click on the last cell in the notebook (containing the text `ROITool(...)` at the end) and press Control+Enter. This should create an output cell below the original cell that contains the annotation tool. The use of this tool is described in the notebook itself. The tool automatically saves your progress whenever you change the subject, hemisphere, or contour, or whenever you press the Save button.
7. Once you have finished annotating (you do not need to be completely finished---just finished for the time being), you can close the Jupyter browser tabs and press Control+C in the terminal window that is running the virtual machine (where you typed `docke compose up --build`) to shut it down.
8. To commit your work to GitHub you will need to can use the following commands (from within the root of the `hcp-annot-vc` repository, where you also ran `docker compose up --build`):

   ```bash
   $ git add save/<username>/*/*
   $ git commit -a -m '<some message>'
   $ git push
   ```
   
   The `<username>` in the first command should be replaced with your GitHub username, and the `<some message>` can be any message you wish to attach to the batch of contours worked on since your last commit. It's fine for this project to commit an empty message (`''`)---comments on individual contours should go in the Notes section of the contour editor. The final command, `git push`, tells Git to sync your work up to your repository on [github.com](https://github.com/). 
   
   **Important**: You should commit and push your work often, ideally after every time you finish working on a set of contours. Pushing the work to GitHub prevents it from being lost.

