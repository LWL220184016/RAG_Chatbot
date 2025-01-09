import torch
from nemo.collections.asr.modules import ConformerEncoder

# Define parameters for the ConformerEncoder
encoder_params = {
    'feat_in': 80,
    'n_layers': 12,
    'd_model': 512,
    'subsampling': 'striding',
    'subsampling_factor': 4,
    'n_heads': 8,
    'ff_expansion_factor': 4,
    'self_attention_model': 'rel_pos',
    'att_context_size': [-1, -1],
    'conv_kernel_size': 31,
    'dropout': 0.1,
}

# Initialize the ConformerEncoder
conformer_encoder = ConformerEncoder(**encoder_params)

# Create a dummy input tensor
# Note: time_steps should be divisible by subsampling_factor for clean subsampling
batch_size = 2
time_steps = 500  # Original time steps
subsampled_time_steps = time_steps // encoder_params['subsampling_factor']  # Adjust for subsampling
features = encoder_params['feat_in']
dummy_input = torch.randn(batch_size, time_steps, features)

# Print information about the dummy input
print("Type of processed audio_data:", type(dummy_input))
print("Class of processed audio_data:", dummy_input.__class__)
print("Original processed_data shape:", dummy_input.shape)

# Pass the dummy input through the encoder
# Note: The length tensor should reflect the original length before subsampling
encoded_output, encoded_lengths = conformer_encoder(
    audio_signal=dummy_input, 
    length=torch.tensor([time_steps]*batch_size)  # Original length before subsampling
)

# Output shapes:
print(f"Encoded output shape: {encoded_output.shape}")
print(f"Encoded lengths: {encoded_lengths}")

# Explanation:
# - The shape of encoded_output should now be (batch_size, subsampled_time_steps, d_model)
# - encoded_lengths will reflect the length after subsampling