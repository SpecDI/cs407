#!/bin/bash
for video in $(find input/videos ! -name "*.md" -type f)
do
    echo "Processing: " $video
    python3 demo.py --sequence_file=$video --enable_cropping --hide_window
    echo "Done"
done
echo "All videos processed."
