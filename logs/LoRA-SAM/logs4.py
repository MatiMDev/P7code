Defaulting to user installation because normal site-packages is not writeable
Looking in indexes: https://pypi.org/simple, https://pypi.ngc.nvidia.com
Requirement already satisfied: PyYAML in /usr/local/lib/python3.12/dist-packages (6.0.2)
Defaulting to user installation because normal site-packages is not writeable
Looking in indexes: https://pypi.org/simple, https://pypi.ngc.nvidia.com
Collecting git+https://github.com/facebookresearch/segment-anything.git
  Cloning https://github.com/facebookresearch/segment-anything.git to /tmp/pip-req-build-q1uhp_t8
  Resolved https://github.com/facebookresearch/segment-anything.git to commit dca509fe793f601edb92606367a655c15ac00fdf
  Preparing metadata (setup.py): started
  Preparing metadata (setup.py): finished with status 'done'
Initializing LoRA-SAM...
Loading datasets...
loading annotations into memory...
Done (t=0.02s)
creating index...
index created!
loading annotations into memory...
Done (t=0.01s)
creating index...
index created!
Setting up training...
Starting training...
Epoch 1: Train Loss = 0.0964, Val Loss = 0.0813
Saved best model with validation loss 0.0813 to ./lora_weights/lora_rank4.safetensors
Epoch 2: Train Loss = 0.0694, Val Loss = 0.0688
Saved best model with validation loss 0.0688 to ./lora_weights/lora_rank4.safetensors
Epoch 3: Train Loss = 0.0538, Val Loss = 0.0698
Epoch 4: Train Loss = 0.0514, Val Loss = 0.0607
Saved best model with validation loss 0.0607 to ./lora_weights/lora_rank4.safetensors
Epoch 5: Train Loss = 0.0443, Val Loss = 0.0548
Saved best model with validation loss 0.0548 to ./lora_weights/lora_rank4.safetensors
Training completed. Best validation loss: 0.0548
Final model saved to ./lora_weights/lora_rank4.safetensors
