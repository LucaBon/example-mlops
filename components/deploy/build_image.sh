#!/bin/bash

VERSION="1.1.0"
REPO="ghcr.io/lucabon/washing-machine-deploy-model"

docker build . -t $REPO:$VERSION
docker push $REPO:$VERSION

docker inspect --format="{{index .RepoDigests 0}}" "$REPO:$VERSION"
