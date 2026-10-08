#!/bin/bash

VERSION="1.1.0"
REPO="kalpa/washing-machine-deploy-model"

sudo docker build . -t $REPO:$VERSION
sudo docker push $REPO:$VERSION

sudo docker inspect --format="{{index .RepoDigests 0}}" "$REPO:$VERSION"
