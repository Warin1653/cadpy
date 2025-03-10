#cadpy

This repository includes code to geometrically processing cadastral data into a format that is optimized for analysis. This involves flattening, merging by attriutes...


##Overview


# Setting up Docker environment
Dockerfile contains all the Python packages needed to run the scripts above. These packages are also listed in requirements.txt. Follow the instructions below to set up the Docker environment.

Using a command line interface, set the current working directory to the project root directory. Build the docker image with: docker build -t ccai-ground-truth .

Once the build has completed, run the docker image with: docker run --rm -p 8888:8888 -v "%cd%/ground-truth":/home/jovyan/work ccai-ground-truth

You should now be able to access the notebook server via a URL.

# Acknowledgements
This dataset was generated through a project funded by the Climate Change AI Innovation Grants Program. We would also like to acknowledge the ml4floods package and Google Earth Engine which provided some functions used to generate this dataset and the Copernicus Emergency Management Service and European Space Agency (ESA; WorldCover 10 m 2020) for providing access to the underlying data.