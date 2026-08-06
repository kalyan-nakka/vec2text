import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
import transformers
from vec2text.experiments import experiment_from_args
from vec2text.run_args import DataArguments, ModelArguments, TrainingArguments


def main():
    # exp = "inverter"
    exp = "corrector"

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Model Arguments
    model_args.max_seq_length = 77
    model_args.model_name_or_path = "google-t5/t5-small" #changed
    model_args.embedder_model_name = "openai/clip-vit-large-patch14" #changed
    # model_args.embedder_dim = 1024  # CLIP-H embedder dim JUST FOR CLIP-ViT-H-14-laion2B-s32B-b79K
    model_args.num_repeat_tokens = 16
    model_args.embedder_no_grad = True
    model_args.use_frozen_embeddings_as_input = True

    # Data Arguments
    data_args.dataset_name = "msmarco" #changed
    data_args.max_eval_samples = 500

    # Training Arguments
    training_args.per_device_train_batch_size = 32
    training_args.per_device_eval_batch_size = 32
    training_args.num_train_epochs = 100
    training_args.eval_steps = 20000
    training_args.warmup_steps = 10000
    training_args.bf16 = True
    training_args.use_wandb = True
    # training_args.experiment = "inversion"
    training_args.experiment = "corrector"
    training_args.lr_scheduler_type = "constant_with_warmup"
    training_args.exp_group_name = ""
    training_args.learning_rate = 0.001
    training_args.output_dir = "./saves/SDV1_4_t5_small_corrector_originalmsmarco_250K_inverter"  # ./saves/{model_name}
    training_args.save_steps = 2000

    if exp == "corrector":
        # Training Arguments
        training_args.output_dir = "./saves/SDV1_4_t5_small_corrector_originalmsmarco_250K_corrector"  # ./saves/{model_name}_corrector
        training_args.corrector_model_alias = "t5-small___CLIP_ViT_L_14__msmarco_original__msl77__10epoch"

    experiment = experiment_from_args(model_args, data_args, training_args)
    experiment.run()


if __name__ == '__main__':
    main()
