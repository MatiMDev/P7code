Defaulting to user installation because normal site-packages is not writeable
Looking in indexes: https://pypi.org/simple, https://pypi.ngc.nvidia.com
Requirement already satisfied: PyYAML in /usr/local/lib/python3.12/dist-packages (6.0.2)
Defaulting to user installation because normal site-packages is not writeable
Looking in indexes: https://pypi.org/simple, https://pypi.ngc.nvidia.com
Collecting git+https://github.com/facebookresearch/segment-anything.git
  Cloning https://github.com/facebookresearch/segment-anything.git to /tmp/pip-req-build-kf4o4sq5
  Resolved https://github.com/facebookresearch/segment-anything.git to commit dca509fe793f601edb92606367a655c15ac00fdf
  Preparing metadata (setup.py): started
  Preparing metadata (setup.py): finished with status 'done'
Initializing LoRA-SAM...
Loading datasets...
loading annotations into memory...
Done (t=0.03s)
creating index...
index created!
loading annotations into memory...
Done (t=0.02s)
creating index...
index created!
Setting up training...
Starting training...
Epoch 1: Train Loss = 0.0954, Val Loss = 0.0775
Saved best model with validation loss 0.0775 to ./lora_weights/lora_rank2.safetensors
Epoch 2: Train Loss = 0.0679, Val Loss = 0.0725
Saved best model with validation loss 0.0725 to ./lora_weights/lora_rank2.safetensors
Epoch 3: Train Loss = 0.0596, Val Loss = 0.0654
Saved best model with validation loss 0.0654 to ./lora_weights/lora_rank2.safetensors
Epoch 4: Train Loss = 0.0510, Val Loss = 0.0665
Epoch 5: Train Loss = 0.0464, Val Loss = 0.0620
Saved best model with validation loss 0.0620 to ./lora_weights/lora_rank2.safetensors
Training completed. Best validation loss: 0.0620
Final model saved to ./lora_weights/lora_rank2.safetensors
