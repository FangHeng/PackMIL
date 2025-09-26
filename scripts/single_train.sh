#!/bin/bash
MODE=$1
shift

handle_wandb_cache() {
    local args=("$@")
    local output_path=""
    local project=""
    local title=""
    
    # Parse arguments to find output_path and project
    for ((i=0; i<${#args[@]}; i++)); do
        case "${args[i]}" in
            --output_path)
                output_path="${args[i+1]}"
                ;;
            --project)
                project="${args[i+1]}"
                ;;
            --title)
                title="${args[i+1]}"
                ;;
        esac
    done

    if [[ -n "$project" && -n "$output_path" && -n "$title" && "${WANDB_MODE}" == "offline" ]]; then
        local source_dir="${output_path}/wandb/latest-run"
        local dest_dir="${output_path}/wandb_cache/${project}/${title}/wandb"
        
        if [[ -d "$source_dir" ]]; then
            local run_id="run_$(date +%Y%m%d_%H%M%S)_${RANDOM}"
            dest_dir="${dest_dir}/${run_id}"
            
            mkdir -p "$(dirname "${dest_dir}")"
            cp -r "${source_dir}" "${dest_dir}"
            echo "Copied wandb files from ${source_dir} to ${dest_dir}"
        fi
    fi
}

if [[ $MODE == "train" ]]; then
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python3 ../main.py "$@"
    # Handle wandb cache after training
    handle_wandb_cache "$@"
elif [[ $MODE == "eval" ]]; then
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python3 ../evaluate.py "$@"
    # Handle wandb cache after evaluation
    handle_wandb_cache "$@"
fi