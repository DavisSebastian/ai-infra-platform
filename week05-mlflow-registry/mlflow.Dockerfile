# The official MLflow image, plus the one thing it lacks: a Postgres driver.
#
# MLflow talks to its database through SQLAlchemy, and SQLAlchemy needs a driver per database
# (like a JDBC driver jar). The official image ships without psycopg2, so a postgresql:// backend
# store fails at startup with "No module named 'psycopg2'". Both versions are pinned so a rebuild
# next month gives the same image.
FROM ghcr.io/mlflow/mlflow:v3.17.2

RUN pip install --no-cache-dir psycopg2-binary==2.9.13
