#!/bin/bash

# Create or clear a file to store job IDs
job_log="job_ids.log"
echo "Recording job IDs in $job_log"
echo "Command | Job ID" > "$job_log"
echo "-----------------" >> "$job_log"

Drop  # Set your dropping position number here

for retention in 0.10 0.30 0.50 0.70 0.90; do
    for sampler in uniform st_gaussian tb; do
        if [[ "$sampler" == "st_gaussian" ]]; then
            # Submit job for Charades and capture job ID
            charades_cmd="sbatch launch_job/run_mcq.sh $retention $sampler Charades 1 $Drop"
            charades_output=$($charades_cmd)
            charades_job_id=$(echo "$charades_output" | awk '{print $NF}')  # Extract the last word (job ID)
            echo "$charades_cmd | $charades_job_id" >> "$job_log"

            # Submit job for PerceptionTest and capture job ID
            perception_cmd="sbatch launch_job/run_mcq.sh $retention $sampler PerceptionTest 1 $Drop"
            perception_output=$($perception_cmd)
            perception_job_id=$(echo "$perception_output" | awk '{print $NF}')  # Extract the last word (job ID)
            echo "$perception_cmd | $perception_job_id" >> "$job_log"

        elif [[ "$sampler" == "uniform" ]]; then
            # Submit job for Charades and capture job ID
            charades_cmd="sbatch launch_job/run_mcq.sh $retention $sampler Charades 0 $Drop"
            charades_output=$($charades_cmd)
            charades_job_id=$(echo "$charades_output" | awk '{print $NF}')  
            echo "$charades_cmd | $charades_job_id" >> "$job_log"

            # Submit job for PerceptionTest and capture job ID
            perception_cmd="sbatch launch_job/run_mcq.sh $retention $sampler PerceptionTest 0 $Drop"
            perception_output=$($perception_cmd)
            perception_job_id=$(echo "$perception_output" | awk '{print $NF}')  
            echo "$perception_cmd | $perception_job_id" >> "$job_log"

        else
            # Submit job for Charades and capture job ID
            charades_cmd="sbatch launch_job/run_mcq.sh $retention $sampler Charades 0.5 $Drop"
            charades_output=$($charades_cmd)
            charades_job_id=$(echo "$charades_output" | awk '{print $NF}')  
            echo "$charades_cmd | $charades_job_id" >> "$job_log"

            # Submit job for PerceptionTest and capture job ID
            perception_cmd="sbatch launch_job/run_mcq.sh $retention $sampler PerceptionTest 0.5 $Drop"
            perception_output=$($perception_cmd)
            perception_job_id=$(echo "$perception_output" | awk '{print $NF}')  
            echo "$perception_cmd | $perception_job_id" >> "$job_log"
        fi
    done
done
