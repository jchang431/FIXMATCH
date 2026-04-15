import argparse
import yaml

from train.train_simclr import SimCLRTrainer
from train.train_supervised import SupervisedTrainer
from utils.data_utils import Config


class Runner:
    def __init__(self, args):
        self.mode = args.mode
        self.device = args.device
        self.checkpoint_dir = args.checkpoint_dir

        # config 로드
        if args.config:
            with open(args.config, "r") as f:
                config_dict = yaml.safe_load(f)
                self.config = Config(config_dict)
        else:
            raise ValueError("Config file must be provided")

    def run(self):
        if self.mode == "pretrain":
            return self._run_simclr()
        elif self.mode == "supervised":
            return self._run_supervised() # add supervised(baseline)
        elif self.mode == "linear":
            return self._run_linear()
        elif self.mode == "inference":
            return self._run_inference()
        else:
            raise ValueError("Invalid mode")

    def _run_simclr(self):
        print("Running SimCLR pretraining...")
        trainer = SimCLRTrainer(
            self.config,
            checkpoint_dir=self.checkpoint_dir,
            device=self.device,
        )
        trainer.train()

    def _run_supervised(self):
        print("Running Supervised baseline...")
        trainer = SupervisedTrainer(
            self.config,
            checkpoint_dir=self.checkpoint_dir,
            device=self.device,
        )
        trainer.train()

    def _run_linear(self):
        print("Running linear evaluation...")
        # TODO
        pass

    def _run_inference(self):
        print("Running inference...")
        # TODO
        pass


def get_args():
    parser = argparse.ArgumentParser(description="Training Runner")

    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        required=True,
        choices=["pretrain", "supervised", "linear", "inference"],
        help="Select training mode",
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to config YAML file",
    )

    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use",
    )

    parser.add_argument(
        "--checkpoint-dir",
        "-x",
        type=str,
        default="./checkpoints",
        help="Checkpoint directory",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    runner = Runner(args)
    runner.run()
