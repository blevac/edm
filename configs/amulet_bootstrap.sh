#! /usr/bin/bash
# this file is used to bootstrap the amulet project by setting result and data storage accounts

export AMLT_PROJECT_STORAGE_ACCOUNT_NAME=${AMLT_PROJECT_STORAGE_ACCOUNT_NAME:-"brettwest"}

# JOB_NAME=$(head -c 32 /dev/urandom | openssl base64 | tr -dc a-zA-Z0-9 | head -c 16)

export AMLT_PROJECT_NAME=${AMLT_PROJECT_NAME:-"travis_edm_training"}
export AMLT_REGION_NAME=${AMLT_REGION_NAME:-"westus2"}
export AMLT_DATA_STORAGE_ACCOUNT_NAME=${AMLT_DATA_STORAGE_ACCOUNT_NAME:-"brettwest"}
# amlt project create $AMLT_PROJECT_NAME $AMLT_PROJECT_STORAGE_ACCOUNT_NAME