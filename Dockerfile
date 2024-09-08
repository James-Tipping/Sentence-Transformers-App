# Use a Miniconda base image
FROM continuumio/miniconda3

# Create the environment and install dependencies available via Conda
RUN conda create -n myenv python=3.11 -y \
    && conda install -n myenv -c conda-forge -y \
        h5py=3.11.0 \
        nltk=3.8.1 \
        numpy=1.26.4 \
        pandas=2.2.2 \
        protobuf=4.25.3 \
        scipy=1.13.1 \
        sentence-transformers=3.0.1 \
        fastapi=0.112.0 \
        pydantic=2.7.4 \
        uvicorn=0.30.1 \
        transformers=4.41.2 \
        google-cloud-storage=2.18.2

# Activate the environment
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# Set the working directory
WORKDIR /app

# Copy the application code
COPY . .

# Run your application using python
CMD ["conda", "run", "-n", "myenv", "python", "main.py"]