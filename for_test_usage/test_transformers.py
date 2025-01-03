from transformers import AutoProcessor, AutoModelForImageTextToText, pipeline
import torch
from PIL import Image


# Load model directly
from transformers import AutoProcessor, AutoModelForImageTextToText

processor = AutoProcessor.from_pretrained("meta-llama/Llama-3.2-11B-Vision-Instruct")
model = AutoModelForImageTextToText.from_pretrained("meta-llama/Llama-3.2-11B-Vision-Instruct").to('cuda')

# Create a pipeline for image-text-to-text generation
imageTextToText = pipeline("image-text-to-text", model=model, processor=processor, device=0)

# Rest of your code remains the same
image_path = "img/cat1.jpg"
image = Image.open(image_path)

result = imageTextToText(image, text="What is this?")

print(result)

# # success to run this for generating text
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# import torch

# # Load model and tokenizer with specified torch dtype
# model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3.5-vision-instruct", trust_remote_code=True, torch_dtype=torch.float16).to('cuda')
# tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-vision-instruct", trust_remote_code=True)

# # Move the model to GPU
# model.to('cuda')

# # Create a pipeline for text generation
# text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

# image_path = "img/cat1.jpg"
# with open(image_path, "rb") as image_file:
#     image_data = image_file.read()

# result = text_generator("What is the capital of France?")

# print(result)