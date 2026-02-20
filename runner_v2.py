import sys

import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
import transformers
import argparse
from vec2text.experiments import experiment_from_args
from vec2text.run_args import DataArguments, ModelArguments, TrainingArguments

def parse_arguments():
    parser = argparse.ArgumentParser("Inverter_Corrector_Trainer")
    # Type of training
    parser.add_argument("--experiment_arg", type=str, default="inverter", choices=["inverter", "corrector"], help="Type of training: inverter or corrector")
    # Embedder model name
    # Replaced longer names for shorter ones:
    # CLIP_V1_4 = openai/clip-vit-large-patch14
    # CLIP_L_V1_4 = laion/CLIP-ViT-L-14-laion2B-s32B-b82K
    # CLIP_H_V1_4 = laion/CLIP-ViT-H-14-laion2B-s32B-b79K
    parser.add_argument("--embedder_model_name_arg", type=str, default="CLIP_V1_4", choices=["CLIP_V1_4", "CLIP_L_V1_4", "CLIP_H_V1_4"], help="Name of the embedder model")
    # Dataset Name
    parser.add_argument("--dataset_name_arg", type=str, default="msmarco_100K", choices=["msmarco_100K", "msmarco_250K", "msmarco_500K", "msmarco_750K", "msmarco_1M", "msmarco_1_25M", "msmarco_1_5M", "msmarco_1_75M", "msmarco_2M"], help="Name of the dataset to use for training the inverter")
    # Output Directory
    parser.add_argument("--output_dir_arg", type=str, default="./saves/UnsavedModel", help="Directory to save the model and logs")
    # Corrector model alias (if corrector)
    parser.add_argument("--corrector_model_alias_arg", type=str, help="Alias of the corrector model to use for training the inverter")


    return parser.parse_known_args()[0]

def main():
    args = parse_arguments()
    custom_flags = ["--experiment_arg", "--embedder_model_name_arg", "--dataset_name_arg", 
                    "--output_dir_arg", "--corrector_model_alias_arg"]
    filtered_argv = [sys.argv[0]]
    skip_next = False
    for arg in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg in custom_flags:
            skip_next = True
            continue
        filtered_argv.append(arg)
    sys.argv = filtered_argv

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Model Arguments
    model_args.max_seq_length = 77
    model_args.model_name_or_path = "google-t5/t5-small"

    if args.experiment_arg == "inverter":
        training_args.experiment = "inversion"
    elif args.experiment_arg == "corrector":
        training_args.experiment = "corrector"
        training_args.corrector_model_alias = args.corrector_model_alias_arg
    

    if args.embedder_model_name_arg == "CLIP_V1_4":
        model_args.embdeder_model_name = "openai/clip-vit-large-patch14"
    elif args.embedder_model_name_arg == "CLIP_L_V1_4":
        model_args.embdeder_model_name = "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
    elif args.embedder_model_name_arg == "CLIP_H_V1_4":
        model_args.embdeder_model_name = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"

    # model_args.embdeder_model_name = args.embedder_model_name
    if args.embedder_model_name_arg == "CLIP_H_V1_4":
        model_args.embedder_dim = 1024
    model_args.num_repeat_tokens = 16
    model_args.embedder_no_grad = True
    model_args.use_frozen_embeddings_as_input = True

    # Data Arguments
    data_args.dataset_name = args.dataset_name_arg
    data_args.max_eval_samples = 500

    # Training Arguments
    training_args.per_device_train_batch_size = 32
    training_args.per_device_eval_batch_size = 32
    training_args.num_train_epochs = 100
    training_args.eval_steps = 20000
    training_args.warmup_steps = 10000
    training_args.bf16 = True
    training_args.use_wandb = True
    training_args.lr_scheduler_type = "constant_with_warmup"
    training_args.exp_group_name = ""
    training_args.learning_rate = 0.001
    training_args.output_dir = args.output_dir_arg
    training_args.save_steps = 2000

    experiment = experiment_from_args(model_args, data_args, training_args)
    experiment.run()


if __name__ == '__main__':
    main()
