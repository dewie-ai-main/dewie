#!/bin/bash
PORT=${1:-10946}
PID=$(lsof -ti tcp:$PORT)
if [ -z "$PID" ]; then
    echo "Nothing running on :$PORT"
else
    kill $PID
    echo "Stopped PID $PID (was on :$PORT)"
fi
