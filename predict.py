from cog import BasePredictor, Input, Path
import os
import time
import torch
import subprocess
from typing import List
import base64
import tempfile
import tarfile
from io import BytesIO
import numpy as np
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from transformers import CLIPTextModel, CLIPTokenizer,T5EncoderModel, T5TokenizerFast
# from optimum.quanto import freeze, qfloat8, quantize
# from optimum.quanto.models.diffusers_models import QuantizedDiffusersModel
# from optimum.quanto.models.transformers_models import QuantizedTransformersModel
from weights import WeightsDownloadCache
dtype = torch.bfloat16
bfl_repo = "./FLUX.1-dev"
class Predictor(BasePredictor):
    def load_trained_weights(
        self, weights: Path | str, pipe: FluxPipeline, lora_scale: float
    ):
        if isinstance(weights, str) and weights.startswith("data:"):
            # Handle data URL
            print("Loading LoRA weights from data URL")
            _, encoded = weights.split(",", 1)
            data = base64.b64decode(encoded)
            with tempfile.TemporaryDirectory() as temp_dir:
                with tarfile.open(fileobj=BytesIO(data), mode="r:*") as tar:
                    tar.extractall(path=temp_dir)
                lora_path = os.path.join(
                    temp_dir, "output/flux_train_replicate/lora.safetensors"
                )
                pipe.load_lora_weights(weight_name=lora_path)
                pipe.fuse_lora(lora_scale=lora_scale)
        else:
            # Handle local path
            print(f"Loading LoRA weights from {weights}")
            local_weights_cache = self.weights_cache.ensure(str(weights))
            lora_path = os.path.join(
                local_weights_cache, "output/flux_train_replicate/lora.safetensors"
            )
            pipe.load_lora_weights(lora_path)

        print("LoRA weights loaded successfully")

    def setup(self) -> None:
        self.weights_cache = WeightsDownloadCache()
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(bfl_repo, subfolder="scheduler")
        text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
        tokenizer_2 = T5TokenizerFast.from_pretrained(bfl_repo, subfolder="tokenizer_2", torch_dtype=dtype)
        vae = AutoencoderKL.from_pretrained(bfl_repo, subfolder="vae", torch_dtype=dtype)
        
        #text_encoder_2 = T5EncoderModel.from_pretrained(bfl_repo, subfolder="text_encoder_2", torch_dtype=dtype)
        #transformer = FluxTransformer2DModel.from_pretrained(bfl_repo, subfolder="transformer", torch_dtype=dtype)

        # quantize(transformer, weights=qfloat8)
        # freeze(transformer)
        # torch.save(transformer, './transformer.pt')
        transformer = torch.load('./transformer.pt')

        # quantize(text_encoder_2, weights=qfloat8)
        # freeze(text_encoder_2)
        # torch.save(text_encoder_2, './text_encoder_2.pt')
        text_encoder_2 = torch.load('./text_encoder_2.pt')



        self.dev_pipe = FluxPipeline(
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=None,
            tokenizer_2=tokenizer_2,
            vae=vae,
            transformer=None,
        )
        self.dev_pipe.text_encoder_2 = text_encoder_2
        self.dev_pipe.transformer = transformer
        self.dev_pipe.enable_model_cpu_offload()
        

    #@torch.amp.autocast("cuda")

    def aspect_ratio_to_width_height(self, aspect_ratio: str):
        aspect_ratios = {
            "1:1": (1024, 1024),
            "16:9": (1344, 768),
            "21:9": (1536, 640),
            "3:2": (1216, 832),
            "2:3": (832, 1216),
            "4:5": (896, 1088),
            "5:4": (1088, 896),
            "9:16": (768, 1344),
            "9:21": (640, 1536),
        }
        return aspect_ratios.get(aspect_ratio)

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(description="Prompt for generated image"),
        aspect_ratio: str = Input(
            description="Aspect ratio for the generated image",
            choices=["1:1", "16:9", "21:9", "2:3", "3:2", "4:5", "5:4", "9:16", "9:21"],
            default="1:1",
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        lora_scale: float = Input(
            description="Determines how strongly the LoRA should be applied. Sane results between 0 and 1.",
            default=1.0,
            le=2.0,
            ge=-1.0,
        ),
        num_inference_steps: int = Input(
            description="Number of inference steps",
            ge=1,
            le=50,
            default=20,
        ),
        model: str = Input(
            description="Which model to run inferences with. The dev model needs around 28 steps but the schnell model only needs around 4 steps.",
            choices=["dev", "schnell"],
            default="dev",
        ),
        guidance_scale: float = Input(
            description="Guidance scale for the diffusion process",
            ge=0,
            le=10,
            default=3.5,
        ),
        seed: int = Input(
            description="Random seed. Set for reproducible generation", default=None
        ),
        output_format: str = Input(
            description="Format of the output images",
            choices=["webp", "jpg", "png"],
            default="webp",
        ),
        output_quality: int = Input(
            description="Quality when saving the output images, from 0 to 100. 100 is best quality, 0 is lowest quality. Not relevant for .png outputs",
            default=80,
            ge=0,
            le=100,
        ),
        replicate_weights: str = Input(
            description="Replicate LoRA weights to use. Leave blank to use the default weights.",
            default=None,
        ),
    ) -> List[Path]:
        """Load the model into memory to make running multiple predictions efficient"""
        start = time.time()

        print("setup took: ", time.time() - start)
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        width, height = self.aspect_ratio_to_width_height(aspect_ratio)
        max_sequence_length = 512

        flux_kwargs = {}
        print(f"Prompt: {prompt}")
        print("txt2img mode")
        flux_kwargs["width"] = width
        flux_kwargs["height"] = height
        if replicate_weights:
            flux_kwargs["joint_attention_kwargs"] = {"scale": lora_scale}
        print("Using dev model")
        pipe = self.dev_pipe
    

        if replicate_weights:
            pipe.load_lora_weights("./Lora.safetensors")
            pipe.fuse_lora(lora_scale=lora_scale)
            #self.load_trained_weights(replicate_weights, pipe, lora_scale)

        generator = torch.Generator("cuda").manual_seed(seed)

        common_args = {
            "prompt": [prompt] * num_outputs,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "num_inference_steps": num_inference_steps,
            "max_sequence_length": max_sequence_length,
            "output_type": "pil",
        }

        output = pipe(**common_args, **flux_kwargs)

        if replicate_weights:
            pipe.unload_lora_weights()

        output_paths = []
        for i, image in enumerate(output.images):
            output_path = f"./out-{i}.{output_format}"
            if output_format != "png":
                image.save(output_path, quality=output_quality, optimize=True)
            else:
                image.save(output_path)
            output_paths.append(Path(output_path))


        return output_paths