FROM python:3.10-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"] 

RUN pipenv install --system --deploy

COPY ["predict.py", "utils.py", "dv_model.bin", "xgb_model.bin", "scaler.bin", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn",  "--bind", "0.0.0.0:9696", "predict:app"]