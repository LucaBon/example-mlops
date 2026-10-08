FROM python:3.9

ARG KUBECTL_VERSION=v1.27.16

# Install kubectl
RUN curl -fsSLo /usr/local/bin/kubectl \
      https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/amd64/kubectl \
    && chmod +x /usr/local/bin/kubectl

COPY requirements.txt .
RUN python3 -m pip install -r \
    requirements.txt --quiet --no-cache-dir \
    && rm -f requirements.txt

ADD . /workspace
WORKDIR /workspace

ENTRYPOINT ["python", "src/deploy.py"]
