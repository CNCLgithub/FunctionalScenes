#!/usr/bin/env bash


sconfig_dir=$(realpath "$0" | xargs dirname)
project_dir="$1"

if [ -z "$project_dir" -a "$project_dir" != " " ]; then
    project_dir=$(dirname "${sconfig_dir}")
    echo "No project directory provided, using ${project_dir}"
fi

echo "Initializing config files"
cp "$sconfig_dir/template_run.sh" "$project_dir/run.sh"
cp "$sconfig_dir/template_setup.sh" "$project_dir/setup.sh"
cp "$sconfig_dir/template_senv.conf" "$project_dir/default.conf"
cp "$sconfig_dir/load_config.sh" "$project_dir/load_config.sh"

# chmod +x "$project_dir/*.sh"
