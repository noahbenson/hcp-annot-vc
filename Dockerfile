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

USER root
# Make a directory for the repos.
RUN mkdir -p /repo && chown $NB_USER /repo
RUN mkdir -p /data && chown $NB_USER /data
# We also want to install R and lme4 for use with our data.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends software-properties-common
RUN wget -q -O /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc \
         https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc \
 && add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu jammy-cran40/" \
 && apt-get update \
 && apt-get install -y --no-install-recommends r-base r-base-dev
RUN R -e 'install.packages("lme4", repos="http://cran.us.r-project.org")'

USER $NB_USER
# We need to install pytorch.
RUN pip install torch
# And we need to put the the hcpannot library inside of this image--both the
# data and the analysis library.
RUN cd /repo \
 && mkdir data \
 && cd data \
 && git clone -b data https://github.com/noahbenson/hcp-annot-vc .
RUN cd /repo \
 && mkdir analysis \
 && cd analysis \
 && git clone -b analysis https://github.com/noahbenson/hcp-annot-vc . \
 && pip install -e .

