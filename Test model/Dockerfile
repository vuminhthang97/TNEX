FROM python:3.8-slim

ARG SCRIPT=predict_job.py

RUN pip install --no-cache s3fs fsspec pandas scikit-learn numpy &&\
    useradd -u 1000 techx &&\
    chown techx /opt

USER techx

COPY --chown=techx:techx ${SCRIPT} /opt/app/run.py

# CMD [ "python" , "/opt/app/${SCRIPT}"]

ENTRYPOINT [ "python" , "/opt/app/run.py"]