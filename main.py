import argparse
import yaml

#from train.train_simclr import SimCLRTrainer
from train.train_supervised import SupervisedTrainer
from utils.data_utils import Config
from train.train_fixmatch import train_fixmatch


class Runner:
    def __init__(self, args):
        self.mode = args.mode
        self.device = args.device
        self.checkpoint_dir = args.checkpoint_dir

        if args.config:
            with open(args.config, "r") as f:
                config_dict = yaml.safe_load(f)
                self.config = Config(config_dict)
        else:
            raise ValueError("Config file must be provided")

    def run(self):
        if self.mode == "pretrain":
            return self._run_simclr()
        elif self.mode == "supervised_loop":
            return self._run_supervised_loop()
        elif self.mode == "fixmatch_loop":
            return self._run_fixmatch_loop()
        elif self.mode == "supervised":
            return self._run_supervised_once()
        elif self.mode == "fixmatch":
            return self._run_fixmatch_once()
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

    def _run_supervised_once(self):
        pct = self.config.data.label_pct

        print("\n" + "=" * 60)
        print(f"Running supervised baseline with {int(pct * 100)}% labeled data")
        print("=" * 60)

        trainer = SupervisedTrainer(
            self.config,
            checkpoint_dir=f"{self.checkpoint_dir}/supervised_{int(pct * 100)}pct",
            device=self.device,
        )

        history, per_class, test_acc, model = trainer.train()

        print("\nPer-class accuracy:")
        for cls_name, acc in per_class.items():
            print(f"  {cls_name:12s}: {acc * 100:.2f}%")

        print(f"\nTest Acc: {test_acc * 100:.2f}%")

        return {
            "label_pct": pct,
            "history": history,
            "per_class": per_class,
            "test_acc": test_acc,
            "model": model,
        }
    
    def _run_supervised_loop(self):
        print("Running supervised baseline for multiple label portions...")
    
        label_portions = [0.1, 0.25, 0.5, 1.0]
        results = []
    
        for pct in label_portions:
            self.config.data.label_pct = pct
            result = self._run_supervised_once()
            results.append(result)
    
        print("\n" + "=" * 60)
        print("Final Summary")
        print("=" * 60)
        for result in results:
            print(f"{int(result['label_pct'] * 100)}% labeled data -> Test Acc: {result['test_acc']:.4f}")
    
        return results

    def _run_fixmatch_once(self):
        pct = self.config.data.label_pct
    
        print("\n" + "=" * 60)
        print(f"Running FixMatch with {int(pct * 100)}% labeled data")
        print("=" * 60)
    
        history, per_class, test_acc, model = train_fixmatch(self.config)
    
        return {
            "label_pct": pct,
            "history": history,
            "per_class": per_class,
            "test_acc": test_acc,
            "model": model,
        }
    
    def _run_fixmatch_loop(self):
        print("Running FixMatch for multiple label portions...")
    
        label_portions = [0.1, 0.25, 0.5, 1.0]
        results = []
    
        for pct in label_portions:
            self.config.data.label_pct = pct
            result = self._run_fixmatch_once()
            results.append(result)
    
        print("\n" + "=" * 60)
        print("Final Summary")
        print("=" * 60)
        for result in results:
            print(f"{int(result['label_pct'] * 100)}% labeled data -> Test Acc: {result['test_acc']:.4f}")
    
        return results

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
        choices=["pretrain", 
                 "linear", 
                 "inference", 
                 "supervised", 
                 "fixmatch",
                 "supervised_loop",
                 "fixmatch_loop"],
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
