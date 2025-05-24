#!/bin/bash

# --- Settings ---
datasets=("mutagenicity")
positional_encodings=("RWPE" "LapPE")
Kmax_list=(1 2 3 4 5)
sample=100
class_select=1
alpha_list="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9"

# Helper for pe_enc_dim
get_pe_enc_dim() {
  if [ "$1" = "RWPE" ]; then
    echo 16
  else
    echo 4
  fi
}

# --- Main model: GTbase_main_... ---
for data_name in "${datasets[@]}"; do
  for pe in "${positional_encodings[@]}"; do
    pe_enc_dim=$(get_pe_enc_dim $pe)
    model_dir="./"
    model_file="${data_name}"
    # EdAM-s and EdAM-i sweeps, with and without popularity adjustment
    for method in "EdAM_s" "EdAM_i"; do
      for Kmax in "${Kmax_list[@]}"; do
        for pop_flag in "--adjust_popularity" ""; do
          python3 evaluation_faithfulness.py \
            --data_name "$model_file" \
            --model_save_dir "$model_dir" \
            --model GTbase \
            --positional_encoding "$pe" \
            --pe_dim 16 \
            --pe_enc_dim "$pe_enc_dim" \
            --channels 32 \
            --num_layers 1 \
            --num_heads 1 \
            --num_classes 2 \
            --explanation_method "$method" \
            --Kmax "$Kmax" \
            --alpha_list "$alpha_list" \
            --class_select $class_select \
            --sample $sample \
            --log_file "./evaluation_faithfulness_GTbase_main.log" \
            $pop_flag \
            --verbose
        done
      done
    done
    # Baselines (no popularity adjustment)
    for method in "random" "naive" "node_averaging"; do
      python3 evaluation_faithfulness.py \
        --data_name "$model_file" \
        --model_save_dir "$model_dir" \
        --model GTbase \
        --positional_encoding "$pe" \
        --pe_dim 16 \
        --pe_enc_dim "$pe_enc_dim" \
        --channels 32 \
        --num_layers 1 \
        --num_heads 1 \
        --num_classes 2 \
        --explanation_method "$method" \
        --Kmax 2 \
        --alpha_list "$alpha_list" \
        --class_select $class_select \
        --sample $sample \
        --log_file "./evaluation_faithfulness_GTbase_main.log" \
        $use_split \
        $wandb_flag \
        --verbose
    done
  done
done


# LapPE: pe_dim in 4 8 16 32, pe_enc_dim always 4
for data_name in "${datasets[@]}"; do
  for pe_dim in 4 8 16 32; do
    model_dir="."
    model_file="${data_name}"
    for method in "EdAM_s" "EdAM_i"; do
      for Kmax in "${Kmax_list[@]}"; do
        for pop_flag in "--adjust_popularity" ""; do
          python3 evaluation_faithfulness.py \
            --data_name "$model_file" \
            --model_save_dir "$model_dir" \
            --model GTbase \
            --positional_encoding LapPE \
            --pe_dim $pe_dim \
            --pe_enc_dim 4 \
            --channels 32 \
            --num_layers 1 \
            --num_heads 1 \
            --num_classes 2 \
            --explanation_method "$method" \
            --Kmax "$Kmax" \
            --alpha_list "$alpha_list" \
            --class_select $class_select \
            --sample $sample \
            --log_file "./evaluation_faithfulness_Positional_embedding_LapPE.log" \
            $pop_flag \
            $use_split \
            $wandb_flag \
            --verbose
        done
      done
    done
    for method in "direct_interpretation"; do
      python3 evaluation_faithfulness.py \
        --data_name "$model_file" \
        --model_save_dir "$model_dir" \
        --model GTbase \
        --positional_encoding LapPE \
        --pe_dim $pe_dim \
        --pe_enc_dim 4 \
        --channels 32 \
        --num_layers 1 \
        --num_heads 1 \
        --num_classes 2 \
        --explanation_method "$method" \
        --Kmax 2 \
        --alpha_list "$alpha_list" \
        --class_select $class_select \
        --sample $sample \
        --log_file "./evaluation_faithfulness_Positional_embedding_LapPE.log" \
        --verbose
    done
  done
done