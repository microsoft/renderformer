#!/bin/bash
huggingface-cli download --repo-type dataset renderformer/renderformer-video-data --local-dir video-data

# Check if video-data directory exists
if [ ! -d "video-data" ]; then
    echo "Error: video-data directory not found!"
    exit 1
fi

# Function to process zip files in a directory
process_directory() {
    local dir=$1
    echo "Processing directory: $dir"
    
    # Loop through all zip files in the directory
    for zipfile in "$dir"/*.zip; do
        # Skip if no zip files found
        if [ ! -f "$zipfile" ]; then
            continue
        fi

        # Get the base name without extension
        dirname=$(basename "$zipfile" .zip)
        
        # Create directory if it doesn't exist
        echo "Creating directory: $dir/$dirname"
        mkdir -p "$dir/$dirname"
        
        # Unzip the file into the directory
        echo "Unzipping $zipfile into $dir/$dirname"
        unzip -q "$zipfile" -d "$dir/$dirname"
        
        echo "Completed unzipping $zipfile"
        echo "----------------------------------------"
    done
}

# Process each subdirectory
process_directory "video-data/animations"
process_directory "video-data/teaser-scenes"
process_directory "video-data/simulations"

echo "All zip files have been processed!"
