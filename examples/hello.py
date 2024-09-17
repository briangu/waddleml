import waddle
import random  # Just for simulating metrics, replace with real model code.

# Initialize a new run with waddle.init, specifying the project name
run = waddle.init(project="hello-world")

# Save model inputs, hyperparameters, and metadata in run.config
config = run.config
config.learning_rate = 0.01
config.batch_size = 32
config.model_type = "simple_CNN"  # Example configuration

# Simulate model training and log metrics
for epoch in range(10):
    # Simulate a training loss (replace with actual model code)
    train_loss = random.uniform(0.8, 0.4)  # Example of loss decreasing

    # Log the loss metric to Waddle
    run.log({"epoch": epoch, "loss": train_loss})

    # Optionally, log other metrics like accuracy, learning rate, etc.
    if epoch % 2 == 0:
        accuracy = random.uniform(0.6, 0.9)  # Simulate accuracy
        run.log({"epoch": epoch, "accuracy": accuracy})

# Once training is done, mark the run as finished
run.finish()