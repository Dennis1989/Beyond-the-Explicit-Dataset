#!/bin/bash

URL="https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key=KEY_HERE"

CONTENT_TYPE_HEADER="Content-Type: application/json"

INPUT_CSV="experiments/splits/explicit/german/test_german_explicit_mechanistic_dehumanization.csv"
OUTPUT_CSV="experiments/splits/explicit/german/predictions/german_explicit_mechanistic_perspective_api_scores.csv"

echo "instance_id,toxicity_score,identity_attack_score" > "$OUTPUT_CSV"

declare -A processed_instances

if [ -f "$OUTPUT_CSV" ]; then
  while IFS= read -r line; do
    processed_instances["$line"]=1
  done < <(tail -n +2 "$OUTPUT_CSV" | cut -d',' -f1)
fi


send_message() {
  local instance_id=$1
  local message_text=$2
  local data=$(jq -n --arg text "$message_text" '{comment: {"text": $text}, languages: ["en"], requestedAttributes: {TOXICITY:{},IDENTITY_ATTACK:{}}}')
  local response=$(curl -s -X POST "$URL" \
    -H "$AUTH_HEADER" \
    -H "$CONTENT_TYPE_HEADER" \
    -d "$data")
  local toxicity_score=$(echo "$response" | jq -r '.attributeScores.TOXICITY.summaryScore.value')
  local identity_attack_score=$(echo "$response" | jq -r '.attributeScores.IDENTITY_ATTACK.summaryScore.value')
  echo "\"$instance_id\",\"$toxicity_score\",\"$identity_attack_score\"" >> "$OUTPUT_CSV"
  echo "\"$instance_id\",\"$message_text\",\"$toxicity_score\",\"$identity_attack_score\""
  sleep 3
}

tail -n +2 "$INPUT_CSV" | while IFS=',' read -r instance_id message_text; do
  if [[ -z "${processed_instances["$instance_id"]}" ]]; then
    send_message "$instance_id" "$message_text"
  else
    echo "Skipping (already processed): $message_text (ID: $instance_id)"
  fi
done
