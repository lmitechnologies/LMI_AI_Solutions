#!/bin/bash

#
# Currently, only support cgroup v1 with docker cgroupfs driver
#

# Get all running container full IDs
container_ids=$(docker ps -q --no-trunc)

# Print the header for the table
printf "%-15s %-35s %-15s %-15s %-15s %-15s %-15s\n" "Container ID" "Name" "Total" "Cache" "Kernel" "RSS" "Swap"

total_total=0 total_cache=0 total_kernel=0 total_rss=0 total_swap=0

# Iterate over each container
for full_id in $container_ids; do
    # Get the short container ID
    short_id=$(echo $full_id | cut -c 1-12)

    # Get container name
    container_name=$(docker inspect --format '{{.Name}}' $full_id | sed "s/^\/\|\'//g")

    # Define the path to the memory stats
    memory_stats_path="/sys/fs/cgroup/memory/docker/$full_id"

    # Get memory stats
    total=$(cat ${memory_stats_path}/memory.usage_in_bytes)
    cache=$(grep -e '\btotal_cache\b' ${memory_stats_path}/memory.stat | awk '{print $2}')
    kernel=$(cat ${memory_stats_path}/memory.kmem.usage_in_bytes)
    rss=$(grep -E '\btotal_rss\b' ${memory_stats_path}/memory.stat | awk '{print $2}')
    swap=$(grep -e '\btotal_swap\b' ${memory_stats_path}/memory.stat | awk '{print $2}')

    # Convert from bytes to MB
    total=$((total / 1024 / 1024))
    cache=$((cache / 1024 / 1024))
    kernel=$((kernel / 1024 / 1024))
    rss=$((rss / 1024 / 1024))
    swap=$((swap / 1024 / 1024))

    # Sum the stats
    total_total=$((total_total + total))
    total_cache=$((total_cache + cache))
    total_kernel=$((total_kernel + kernel))
    total_rss=$((total_rss + rss))
    total_swap=$((total_swap + swap))
    
    # Print the details in a formatted row
    printf "%-15s %-35s %-15s %-15s %-15s %-15s %-15s\n" "$short_id" "$container_name" "$total" "$cache" "$kernel" "$rss" "$swap"
done

# Print the total in a formatted row
printf "%-15s %-35s %-15s %-15s %-15s %-15s %-15s\n" "TOTAL" "" "$total_total" "$total_cache" "$total_kernel" "$total_rss" "$total_swap"
