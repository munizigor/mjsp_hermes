from="."
to="hermes-asr-vscode:/home/pita/triton-whisperer"

rsync -zz -zarv --prune-empty-dirs \
    --exclude="testing_bench/results/*" --exclude="*.csv" --exclude="*.xlsx" \
    --exclude="logs/*" --exclude="input/*" --exclude="*.pyc" --exclude="*.zip" \
    --exclude=".git/*" --exclude=".vllm_cache/*" --exclude="hf_models/*" \
    --exclude="*.log" --exclude="hf_cache/*" "$from" "$to"