#!/bin/bash

input_folder="$1"
output_folder="$2"
counter=1
num_iterations=20

# make a log file in the output folder if it doesn't exist
if [ ! -f "$output_folder/log.txt" ]; then
    touch "$output_folder/log.txt"
    # write the input folder path to the log file
    echo "Input folder: $input_folder" > "$output_folder/log.txt"
    # write a line that says encoded images = 0
    echo "Encoded images: 0" >> "$output_folder/log.txt"
fi

for i in $(seq 1 $num_iterations); do
    # Iterate over each image in the input folder
    # Iterate over each image randomly
    for image_path in $(ls -1 $input_folder | shuf); do
        # get the path to a random .jpg file within the image path
        image_path=$(find "$input_folder/$image_path" -type f -name "*.jpg" | shuf | head -n 1)

        # get the number of encoded images from the log file
        counter=$(grep "Encoded images" "$output_folder/log.txt" | cut -d' ' -f3)

        # Get the parent folder name
        parent_folder=$(basename "$(dirname "$input_folder/$image_path")")
        # parent folder name is <number>.<category>, so we need to extract the category
        category=$(echo $parent_folder | cut -d'.' -f2)

        # check if image path already exists in the log file
        if grep -q "$image_path" "$output_folder/log.txt"; then
            echo "Image path: $input_folder/$image_path already exists in the log file"
            continue
        fi

        # Create a subfolder in the output folder with the name <parent_folder>_<counter>
        subfolder_name="${category}_${counter}"
        subfolder_path="$output_folder/$subfolder_name"
        mkdir -p "$subfolder_path"

        # Print the image path and parent folder name
        echo "Image path:$image_path"
        echo "category: $category"
        echo "output folder: $subfolder_path"

        # run clipasso script on input image
        # set working directory to the repo directory
        cd /home/etaisella/repos/CLIPasso
        python /home/etaisella/repos/CLIPasso/run_object_sketching.py --target_file "$image_path" --output_path "$subfolder_path" --num_sketches 1

        # delete every file in the output folder except for 'best_iter.jpg', 'best_iter.svg', 'input.png'
        find "$subfolder_path" -type f ! -name 'best_iter.jpg' ! -name 'best_iter.svg' ! -name 'input.png' -delete

        # Increment the counter
        ((counter++))

        # update the log file
        echo "Image path: $image_path" >> "$output_folder/log.txt"

        # update the encoded images count in log file
        sed -i "s/Encoded images: $((counter-1))/Encoded images: $counter/" "$output_folder/log.txt"

    done 
done