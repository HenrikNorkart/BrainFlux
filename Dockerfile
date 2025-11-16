# FROM nvcr.io/nvidia/pytorch:25.03-py3
FROM vllm/vllm-openai:latest


# Set an argument for the frontend, making it clear this is for build-time
ARG DEBIAN_FRONTEND=noninteractive
# Set the timezone to a default value like UTC to prevent prompts
ENV TZ=Etc/UTC


WORKDIR /workspace
COPY . .


RUN python3 -m pip install --upgrade pip \
 && python3 -m pip install -e .

# In your Dockerfile
# RUN apt-get update && apt-get install -y python3-tk

ENTRYPOINT ["/bin/bash", "-lc"]
CMD ["python3 -u /workspace/rogue_one_main.py || tail -f /dev/null"]