#!/bin/bash

# Usage:
# ./test_summarize.sh "your query here"

SERVICE_URL="https://fastapi-test-730472964960.us-central1.run.app"  # replace with your Cloud Run URL
QUERY="$1"

if [ -z "$QUERY" ]; then
  echo "Usage: $0 \"your query here\""
  exit 1
fi

curl -s -X POST "$SERVICE_URL/summarize" \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"$QUERY\"}" | jq
