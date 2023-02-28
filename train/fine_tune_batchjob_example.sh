#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -p dgx-spa,gpu-nvlink
#SBATCH --time=02-17:59:00
#SBATCH -J acc_adapt_ge_largesynth
#SBATCH --mem=20G
#SBATCH --cpus-per-task=8
#SBATCH --output=/data/accent_adaptation/models/fine_tune_wav2vec2_base_ft_960h_German_synth_largesynth_20ep.out

module purge
module load sox/14.4.2
module load anaconda/2021-04-tf2
module load cuda/11.4.0
module load gcc/8.4.0

python ./run_asr_accent_adapt.py \
--output_dir="/data/accent_adaptation/models/wav2vec2-base-ft-960h-German-synth-largesynth-20ep" \
--cache_dir="/data/accent_adaptation/data/cache" \
--num_train_epochs="20" \
--per_device_train_batch_size="96" \
--per_device_eval_batch_size="96" \
--gradient_accumulation_steps="1" \
--save_total_limit="10" \
--evaluation_strategy="epoch" \
--logging_strategy="epoch" \
--save_strategy="epoch" \
--load_best_model_at_end="True" \
--metric_for_best_model="wer" \
--greater_is_better="False" \
--learning_rate="1e-4" \
--warmup_ratio="0.25" \
--model_name_or_path="facebook/wav2vec2-base-960h" \
--target_feature_extractor_sampling_rate \
--dataset_name="accent_adapt_cv_synth_largesynth_German" \
--train_split_name="train" \
--validation_split_name="train" \
--orthography librispeech \
--preprocessing_num_workers="$(nproc)" \
--group_by_length \
--freeze_feature_extractor \
--verbose_logging \
--gradient_checkpointing \
--fp16 \
--half_precision_backend="cuda_amp" \