import hydra
from omegaconf import DictConfig
from src.pipeline import DataPipeline

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    pipeline = DataPipeline(cfg)
    pipeline.run()

if __name__ == "__main__":
    main()