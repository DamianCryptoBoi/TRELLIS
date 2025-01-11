import torch
import os
from diffusers import FluxTransformer2DModel, FluxPipeline, BitsAndBytesConfig, GGUFQuantizationConfig
from transformers import T5EncoderModel, BitsAndBytesConfig as BitsAndBytesConfigTF

class FluxModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
        self.dtype = torch.bfloat16
        self.file_url = "https://huggingface.co/gokaygokay/flux-game/blob/main/hyperflux_00001_.q8_0.gguf"
        self.file_url = self.file_url.replace("/resolve/main/", "/blob/main/").replace("?download=true", "")
        self.single_file_base_model = "camenduru/FLUX.1-dev-diffusers"
        self.quantization_config_tf = BitsAndBytesConfigTF(load_in_8bit=True, bnb_8bit_compute_dtype=torch.bfloat16)
        self.text_encoder_2 = T5EncoderModel.from_pretrained(
            self.single_file_base_model,
            subfolder="text_encoder_2",
            torch_dtype=self.dtype,
            config=self.single_file_base_model,
            quantization_config=self.quantization_config_tf,
            token=self.huggingface_token
        )
        self.transformer = self.load_transformer()
        self.flux_pipeline = FluxPipeline.from_pretrained(
            self.single_file_base_model,
            transformer=self.transformer,
            text_encoder_2=self.text_encoder_2,
            torch_dtype=self.dtype,
            token=self.huggingface_token
        )
        self.flux_pipeline.to(self.device)

    def load_transformer(self):
        if ".gguf" in self.file_url:
            return FluxTransformer2DModel.from_single_file(
                self.file_url,
                subfolder="transformer",
                quantization_config=GGUFQuantizationConfig(compute_dtype=self.dtype),
                torch_dtype=self.dtype,
                config=self.single_file_base_model
            )
        else:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                token=self.huggingface_token
            )
            return FluxTransformer2DModel.from_single_file(
                self.file_url,
                subfolder="transformer",
                torch_dtype=self.dtype,
                config=self.single_file_base_model,
                quantization_config=quantization_config,
                token=self.huggingface_token
            )

    def get_pipeline(self):
        return self.flux_pipeline
    

# Example usage
if __name__ == "__main__":
    flux_model = FluxModel()
    flux_pipeline = flux_model.get_pipeline()
    # Now you can use flux_pipeline for further processing