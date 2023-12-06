import logging
import wandb
import numpy as np

def push_artifacts_table(epoch, loss, r1, r2, predictions):
    """ Returns a wandb.Table object containing all the artifacts
        in the run
    """
    r1 = np.mean(r1)
    r2 = np.mean(r2)
    text_table = wandb.Table(columns=["epoch", "loss", "Rouge1", "Rouge2", "document", "target", "prediction"])

    num_examples = 4
    if len(predictions["document"]) < num_examples:
        num_examples = len(predictions["document"])

    for i in range(num_examples):
        document_i = predictions['document'][i]
        target_i = predictions['target'][i]
        prediction_i = predictions['prediction'][i]

        text_table.add_data(epoch, loss, r1, r2, document_i, target_i, prediction_i)
    wandb.run.log({'Training_Samples' : text_table})