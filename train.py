import torch
import torch.optim as optim

# Your previous code...

# Update the line with ReduceLROnPlateau scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)  # Removed verbose=True

# Your following code...