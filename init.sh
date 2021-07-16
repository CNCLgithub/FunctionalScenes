#!/usr/bin/env bash

sconfig_dir=$(realpath "$0" | xargs dirname)
project_dir="$1"

echo "(  ) Initializing config files"
if [ -z "$project_dir" -a "$project_dir" != " " ]; then
    project_dir=$(dirname "${sconfig_dir}")
    printf "\tNo project directory provided, using %s" "${project_dir}"
fi

cp "${sconfig_dir}/templates/*.sh" "${project_dir}/"
cp "${sconfig_dir}/templates/*.conf" "${project_dir}/"
printf "(\xE2\x9C\x94) Initializing config files\n"

# chmod +x "$project_dir/*.sh"
