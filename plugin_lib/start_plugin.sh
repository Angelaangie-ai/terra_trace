#!/bin/bash
# Start the data process
plugin_data_scripts > /proc/1/fd/1 2>/proc/1/fd/2 &
status=$?
if [ $status -ne 0 ]; then
  echo "Failed to start data_prep.py: $status"
  exit $status
fi
# Start the API server
PORT=${PLUGIN_PORT:-8000}
uvicorn plugin_lib.scripts.main:app --port $PORT --host 0.0.0.0