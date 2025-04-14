FROM apache/airflow:2.9.3-python3.10

USER root
RUN apt-get update && \
    apt-get install -y openjdk-17-jdk && \
    echo "JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64" >> /etc/environment && \
    echo "export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64" >> /etc/profile && \
    echo "export PATH=\$JAVA_HOME/bin:\$PATH" >> /etc/profile

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

USER airflow

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
