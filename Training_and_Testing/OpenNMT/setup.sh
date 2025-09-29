#!/bin/bash

# Set base directory here
base_dir="Code"

# Automatically extract the prefix (S1, S2, S3)
prefix="${base_dir:0:2}"

# Define all possible steps
all_simple_steps=("S1_P7" "S1_P6" "S1_P5" "S1_P4" "S1_P3" "S1_P2" "S1_P1"
                  "S2_P4" "S2_P3" "S2_P2" "S2_P1"
                  "S3_P4" "S3_P3" "S3_P2" "S3_P1")

all_composite_steps=(
  "S1_P1plusP3" "S1_P1plusP4" "S1_P1plusP4plusP5" "S1_P1plusP4plusP6" "S1_P1plusP6" "S1_P1plusP7"
  "S2_P1plusP3" "S2_P1plusP4"
  "S3_P1plusP3" "S3_P1plusP4"
)

# Filter steps by prefix
simple_steps=()
for step in "${all_simple_steps[@]}"; do
  [[ "$step" == $prefix* ]] && simple_steps+=("$step")
done

composite_steps=()
for step in "${all_composite_steps[@]}"; do
  [[ "$step" == $prefix* ]] && composite_steps+=("$step")
done

# Create folders for simple steps (LinearPipeline)
for step in "${simple_steps[@]}"; do
  for level in "Project" "Time"; do
    dir="${step}_${level}Level"
    if [[ -d "$dir" ]]; then
      echo "⏭️  Skipping $dir (already exists)"
    else
      echo "📁 Creating $dir (LinearPipeline)"
      mkdir -p "$dir"
      cp -r "$base_dir/"* "$dir/"

      # 🔧 Update run.sh inside the new folder with folder-specific jobname
      if [[ -f "$dir/run.sh" ]]; then
        echo "🔧 Updating run.sh jobname in $dir"
        sed -i "s/^jobname=.*/jobname=\"OpenNMT_${dir}\"/" "$dir/run.sh"
      else
        echo "⚠️  run.sh not found in $dir"
      fi
    fi
  done
done

# Create folders for composite steps (IsolatedSteps)
for step in "${composite_steps[@]}"; do
  for level in "Project" "Time"; do
    dir="${step}_${level}Level"
    if [[ -d "$dir" ]]; then
      echo "⏭️  Skipping $dir (already exists)"
    else
      echo "📁 Creating $dir (IsolatedSteps)"
      mkdir -p "$dir"
      cp -r "$base_dir/"* "$dir/"

      # 🔧 Update run.sh inside the new folder with folder-specific jobname
      if [[ -f "$dir/run.sh" ]]; then
        echo "🔧 Updating run.sh jobname in $dir"
        sed -i "s/^jobname=.*/jobname=\"OpenNMT_${dir}\"/" "$dir/run.sh"
      else
        echo "⚠️  run.sh not found in $dir"
      fi
    fi
  done
done

echo "✅ Folder setup completed for prefix $prefix."
