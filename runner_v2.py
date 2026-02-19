import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
import transformers
import argparse
from vec2text.experiments import experiment_from_args
from vec2text.run_args import DataArguments, ModelArguments, TrainingArguments

def parse_arguments():
    parser = argparse.ArgumentParser("Inverter_Corrector_Trainer")
    # Type of training
    parser.add_argument("--experiment", type=str, default="inverter", choices=["inverter", "corrector"], help="Type of training: inverter or corrector")
    # Embedder model name
    parser.add_argument("--embedder_model_name", type=str, default="openai/clip-vit-large-patch14", choices=["openai/clip-vit-large-patch14", "laion/CLIP-ViT-L-14-laion2B-s32B-b82K", "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"], help="Name of the embedder model")
    # Dataset Name
    parser.add_argument("--dataset_name", type=str, default="msmarco_100K", choices=["msmarco_100K", "msmarco_250K", "msmarco_500K", "msmarco_750K", "msmarco_1M", "msmarco_1_25M", "msmarco_1_5M", "msmarco_1_75M", "msmarco_2M"], help="Name of the dataset to use for training the inverter")
    # Output Directory
    parser.add_argument("--output_dir", type=str, default="./saves/UnsavedModel", help="Directory to save the model and logs")
    # Corrector model alias (if corrector)
    parser.add_argument("--corrector_model_alias", type=str, help="Alias of the corrector model to use for training the inverter")


    return parser.parse_args()

def main():
    args = parse_arguments()

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # return -1


if __name__ == '__main__':
    main()
