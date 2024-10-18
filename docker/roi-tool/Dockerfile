# This Dockerfile constructs a docker image that contains an installation
# of the Neuropythy library for use with the HCP-annotation project.
#
# Example build:
#   docker build --no-cache --tag nben/hcp-annot-vc:20230329 "$PWD"
#
# Rather than building this dockerfile, it is suggested that users stick
# to using `docker-compose up`. This command loads a prebuilt docker image
# from dockerhub, which is typically faster, and is less likely to break
# as software versions gradually drift.
#
# This Dockerfile has been updated from its original version in order to
# keep future builds relatively stable. The original build instructions
# are available in the GitHub history and remain in the comments below.


# Start with Jupyter scipy-notebook, which includes a lot of what we need.
FROM jupyter/scipy-notebook:python-3.10.6

# Note the Maintainer.
MAINTAINER Noah C. Benson <nben@uw.edu>

# Switch to to the "jovyan" user.
USER $NB_USER

# The original docker was built using these commands; however, once a
# stable configuration was found, the environment was exported, and it
# can now be used with conda.
#   # Install some stuff...
#   RUN conda update --yes -n base conda \
#    && conda install --yes py4j nibabel s3fs
#   RUN conda install --yes -c conda-forge ipywidgets pip
#   RUN pip install --upgrade setuptools
#   RUN pip install 'ipyvolume>=0.5.1'
COPY environment.yml /home/$NB_USER/environment.yml
RUN mamba env update --file /home/$NB_USER/environment.yml --prune

# The original docker was built by installing neuropythy from github by hand;
# this is now installed automatically by the two lines above. Several other
# things below are also handled this way.
#     RUN git clone https://github.com/noahbenson/neuropythy \
#      && cd neuropythy \
#      && pip install -r requirements.txt \
#      && pip install matplotlib \
#      && python setup.py install
#     
#     
#     # Install collapsible cell extensions...
#     
#     # The following lines are a bug workaround; they should be replaced by
#     # just this line once the bug is fixed:
#     
#     # RUN conda install -c conda-forge jupyter_contrib_nbextensions
#     RUN conda install -c conda-forge \
#           'jupyter_contrib_nbextensions < 0.7' 'traitlets == 5.9.0'
#     RUN jupyter contrib nbextension install --user
#     RUN jupyter-nbextension enable collapsible_headings/main \
#      && jupyter-nbextension enable select_keymap/main

# We need to do a few things as root.
USER root

# Install curl so that we can get a few resource files.
RUN apt-get update && apt-get install --yes curl
# Make some directories
RUN mkdir /data \
 && mkdir /save \
 && chown $NB_USER /data /save \
 && chmod 755 /data /save

# Most of the rest can be run as the user.
USER $NB_USER

# Now with curl installed, install the helvetica neue font (for figures).
RUN mkdir -p ~/.local/share/fonts/helvetica_neue_tmp
RUN curl -L -o ~/.local/share/fonts/helvetica_neue_tmp/helveticaneue.zip \
     https://github.com/noahbenson/neuropythy/wiki/files/helveticaneue.zip
RUN cd ~/.local/share/fonts/helvetica_neue_tmp \
 && unzip helveticaneue.zip \
 && mv *.ttf .. \
 && cd .. \
 && rm -r ~/.local/share/fonts/helvetica_neue_tmp
RUN fc-cache -f -v
RUN rm -r ~/.cache/matplotlib

# Remove the work directory
RUN rmdir /home/$NB_USER/work
# The original dockerfile hid neuropythy here, but this is no longer required.
#     RUN mv /home/$NB_USER/neuropythy /home/$NB_USER/.neuropythy

# Reconfigure Jupyter a bit
RUN mkdir -p ~/.jupyter/custom
COPY custom.css /home/$NB_USER/.jupyter/custom/
COPY custom.js /home/$NB_USER/.jupyter/custom/
COPY ipython-startup.py /home/$NB_USER/.ipython/profile_default/startup/
COPY ipython_kernel_config.py /home/$NB_USER/.ipython/profile_default/
COPY roi-drawing.ipynb /home/$NB_USER/open_me.ipynb

# Make sure we have a place to put the hcpannot library.
RUN LPP="`python -c 'import site; print(site.getusersitepackages())'`" \
 && mkdir -p "$LPP" \
 && cd "$LPP" \
 && ln -s "$HOME"/.hcpannot-ext ./hcpannot

# Put our Jupyter settings in place.
COPY jupyter_notebook_config.py /home/$NB_USER/.jupyter/

# Fix some permissions.    
USER root
RUN chown -R $NB_USER /home/$NB_USER/.ipython && chmod 700 /home/$NB_USER/.ipython
USER $NB_USER

# And mark the entrypoint
ENTRYPOINT ["tini", "-g", "--", "/usr/local/bin/start-notebook.sh"]
