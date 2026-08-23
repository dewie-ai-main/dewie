#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/stop_dev_server.sh"
sleep 1
exec "$SCRIPT_DIR/start_dev_server.sh"
