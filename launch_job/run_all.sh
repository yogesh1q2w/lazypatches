#!/bin/bash

# Create or clear a file to store job IDs
job_log="job_ids.log"
echo "Recording job IDs in $job_log"
echo "Command | Job ID" > "$job_log"
echo "-----------------" >> "$job_log"

for fps in 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0; do
    # Submit job for Charades and capture job ID
    charades_cmd="sbatch launch_job/run_mcq.sh $fps 0 None Charades 0 0"
    charades_output=$($charades_cmd)
    charades_job_id=$(echo "$charades_output" | awk '{print $NF}')  # Extract the last word (job ID)
    echo "$charades_cmd | $charades_job_id" >> "$job_log"

    # Submit job for PerceptionTest and capture job ID
    perception_cmd="sbatch launch_job/run_mcq.sh $fps 0 None PerceptionTest 0 0"
    perception_output=$($perception_cmd)
    perception_job_id=$(echo "$perception_output" | awk '{print $NF}')  # Extract the last word (job ID)
    echo "$perception_cmd | $perception_job_id" >> "$job_log"
done

echo "All jobs submitted. Job IDs saved in $job_log"