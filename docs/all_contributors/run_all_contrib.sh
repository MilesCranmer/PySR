#!/bin/bash

yarn install --frozen-lockfile
yarn all-contributors $@
