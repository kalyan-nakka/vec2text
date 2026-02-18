import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
import transformers
import argparse
from vec2text.experiments import experiment_from_args
from vec2text.run_args import DataArguments, ModelArguments, TrainingArguments

def parse_arguments():
    parser = argparse.ArgumentParser("Inverter_Corrector_Trainer")
    # Type of training
   
    # Embedder model name
    # Embedder Dimension
    # Dataset Name
    # Output Directory
    # Corrector model alias (if corrector)


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
