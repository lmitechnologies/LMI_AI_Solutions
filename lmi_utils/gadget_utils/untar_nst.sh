#!/bin/bash

# untar inspection event file (keeping the event_ID in the filename)
# decompress .zst files
# create new files named:
# event_ID.intensity.tiff
# event_ID.profile.tiff

tars_folder=$1

if [ -z "$tars_folder" ]; then
    echo "Usage:"
    echo './untar_nst.sh "your/tar/or/tar.gz/dir"'
    echo 'for example:'
    echo './untar_nst.sh /home/my/Downloads/gocator_0'
    exit 1
fi

for tar_path in $(find "$tars_folder" -name "*.tar" -o -name "*.tar.gz"); do

    dirname="$(dirname $tar_path)"

    basename="$(basename $tar_path)"
    IFS='.' read -ra names <<< "$basename"
    basename="${names[0]}"

    untarred_folder="$dirname/$basename"

    mkdir -p "$untarred_folder"

    tar -xvf "$tar_path" --directory "$untarred_folder"

    echo ==================================
    IFS=$'\n'; set -f
    for f in $(find "$untarred_folder" -name *.tiff.zst); do

        unzipped_tiff_target="${f%.*}"
        basename="$(basename $unzipped_tiff_target)"
        dirname="$(dirname $unzipped_tiff_target)"
        unzipped_tiff_target="$dirname.$basename"

        unzstd -f "$f" -o "$unzipped_tiff_target"
        if [ $? -ne 0 ]; then
            echo "FAIL"
            exit 1
        fi
        ls -al "$unzipped_tiff_target"
    done
    unset IFS; set +f
done