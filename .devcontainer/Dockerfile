FROM nvcr.io/nvidia/pytorch:24.06-py3 


# Set an argument for the frontend, making it clear this is for build-time
ARG DEBIAN_FRONTEND=noninteractive
# Set the timezone to a default value like UTC to prevent prompts
ENV TZ=Etc/UTC


# In your Dockerfile
RUN apt-get update &&apt-get install -y python3-tk



