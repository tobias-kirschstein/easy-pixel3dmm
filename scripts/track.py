from omegaconf import OmegaConf

from pixel3dmm import env_paths
from pixel3dmm.tracking.tracker import Tracker


def main(cfg):
    tracker = Tracker(cfg)
    tracker.run()

if __name__ == '__main__':
    base_conf = OmegaConf.load(f'{env_paths.CODE_BASE}/configs/tracking.yaml')

    cli_conf = OmegaConf.from_cli()
    cfg = OmegaConf.merge(base_conf, cli_conf)

    #os.makedirs('/home/giebenhain/debug_wandb_p3dmm/', exist_ok=True)
    #wandb.init(
    #   dir='/home/giebenhain/debug_wandb_p3dmm/',
    #   #config=config,
    #   project='face-tracking-p3dmm',
    #   #tags=wandb_tags,
    #   #name=cfg.config_name,
#
    #)
    main(cfg)