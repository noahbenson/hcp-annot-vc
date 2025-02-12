# This Dockerfile builds the analysis environment for the HCP visual cortex
# annotation project.

# We start with scipy notebook.
FROM jupyter/scipy-notebook:python-3.10.6

# Note the Maintainer.
MAINTAINER Noah C. Benson <nben@uw.edu>

# Switch to to the "jovyan" user.
USER $NB_USER

# We start by building a conda environment. We use mamba because it's faster.
COPY environment.yml /opt/conda/analysis_base.env.yml
RUN mamba env update --file /opt/conda/analysis_base.env.yml

# That's all!
