import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from typing import List, Tuple

import torch
import hydra
import gradio as gr
from omegaconf import DictConfig

from src import utils

log = utils.get_pylogger(__name__)

def demo(cfg: DictConfig) -> Tuple[dict, dict]:
    """Demo function.
    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    assert cfg.ckpt_path

    log.info("Running Demo")

    log.info(f"Instantiating scripted model <{cfg.ckpt_path}>")
    model = torch.jit.load(cfg.ckpt_path)

    log.info(f"Loaded Model: {model}")

    def recognize_cifar_image(image):
        if image is None:
            return None
        image = torch.tensor(image[None, None, ...], dtype=torch.float32)
        preds = model.forward_jit(image)
        preds = preds[0].tolist()
        return {str(i): preds[i] for i in range(10)}

    im = gr.Image(type="pil")

    demo = gr.Interface(
        fn=recognize_cifar_image,
        inputs=[im],
        outputs=[gr.Label(num_top_classes=10)],
        examples=["horse.jpeg", "bird.jpeg"]
    )

    demo.launch(share=True)

@hydra.main(version_base="1.3", config_path="../configs", config_name="demo_scripted.yaml")
def main(cfg: DictConfig) -> None:
    demo(cfg)

if __name__ == "__main__":
    main()