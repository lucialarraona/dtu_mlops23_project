# Base image # with GPU support
FROM  nvcr.io/nvidia/pytorch:22.07-py3 
# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*


WORKDIR /mlops_project

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install "dvc[gs]" 
RUN dvc init --no-scm
RUN dvc remote add -d storage gs://mlops-project-data-44/
RUN dvc pull -f
RUN ls -al /mlops_project


COPY data.dvc data.dvc
COPY setup.py setup.py
COPY src/ src/
COPY models/ models/

ENTRYPOINT ["python", "-u", "src/models/train_model.py"]