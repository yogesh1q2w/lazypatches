from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, AutoTokenizer

# Define the model name and the local directory to save the model
model_name = "Qwen/Qwen2-VL-7B-Instruct"
save_directory = "/home/atuin/g102ea/shared/group_10/models"  # Set your desired path

# Download and save the model, tokenizer, and processor
model = Qwen2VLForConditionalGeneration.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)

# Save locally
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)
processor.save_pretrained(save_directory)
