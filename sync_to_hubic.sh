
#! /bin/sh

if [ ! -d "$1" ]; then
    echo Usage: $0 subdirectory
    exit 1
fi

for SUBDIR in "$@" 
do
    if [ -d "${SUBDIR}" ]; then
	rclone --checksum sync "${SUBDIR}" "hubic:default/Christophe_Pallier/LePetitPrince_2018/${SUBDIR}" 
    fi
done
