FROM ucsdets/datascience-notebook:2020.2-stable

USER root

RUN pip install --no-cache-dir scikit-learn

USER $NB_UID