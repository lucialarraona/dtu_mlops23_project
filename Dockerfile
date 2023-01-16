# Base image # with GPU support
FROM  nvcr.io/nvidia/pytorch:22.07-py3 

WORKDIR /mlops_project
# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
# Only one run command, more efficient
RUN pip install -r requirements.txt --no-cache-dir && \
    pip install get-project-root wandb "dvc[gs]" && \
    dvc init --no-scm && \
    dvc remote add -d storage gs://mlops-project-data-44/ && \
    dvc pull -f

#For debugging, not in final dockerfile
RUN ls -al /mlops_project

# Copy folders necessary
COPY data.dvc data.dvc
COPY setup.py setup.py
COPY src/ src/
COPY models/ models/

ENTRYPOINT ["python", "-u", "src/models/train_model.py"]