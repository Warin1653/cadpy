# use Python 3.10 image: https://github.com/jupyter/docker-stacks#using-old-images
FROM quay.io/jupyter/scipy-notebook:4d70cf8da953

# add geospaital and visualisation packages
RUN mamba install --quiet --yes \
    'geopandas' \ 
    'shapely' \
    'rtree' \
    'folium' \
    'pysal' \
    'pyarrow' \
    'pygeos' \
    'pyogrio' \
    'dask=2025.2.0' \
    'dask-geopandas' && \
    mamba clean --all -f -y && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

# https://github.com/geopandas/geopandas/issues/2442
ENV PROJ_LIB=/opt/conda/share/proj 