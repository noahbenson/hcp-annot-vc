# This Dockerfile constructs a docker image that contains an installation
# of the Neuropythy library for use with the HCP-annotation project.
#
# Example build:
#   docker build --no-cache --tag hcp-annot-vc:analysis "$PWD"
#
#   (but really, use docker-compose up instead).
#

# Start with the hcp-annot-vc image build for annotation.
FROM nben/hcp-annot-vc:20230329

USER $NB_USER

# For the analysis version, we don't want the settings we installed previously;
# they are mostly for making Jupyter look like an app instead of a notebook.
RUN rm -rf /home/$NB_USER/.jupyter
# Also delete the old hcpannot library--we're going to put the current version
# instide the docker image instead.
RUN rm -rf "$HOME"/hcpannot
# Also delete the open_me.ipynb notebook.
RUN rm -rf "$HOME"/open_me.ipynb

# We need to install pytorch.
RUN pip install torch

# Put the hcpannot library inside of this image and install it.
RUN git clone -b analysis https://github.com/noahbenson/hcp-annot-vc repo \
 && cd repo \
 && pip install -e .

# We also want to install lme4 for use with our data.
RUN R -e 'install.packages("lme4", repos="http://cran.us.r-project.org")'

# We also want to check out the current state of the data repo and link it
# to the appropriate rater IDs in the data directory.
USER root
RUN mkdir -p /data /data_branch \
    && chown $NB_USER /data \
    && chown $NB_USER /data_branch

USER $NB_USER
RUN cd /data_branch \
 && git clone -b data https://github.com/noahbenson/hcp-annot-vc .

