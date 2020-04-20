import os
import torch

from models.cvae import CVAE

## Evaluation
def evaluate_lower_bound(model, labeled_test_subset):
    check_model = isinstance(model, CVAE)
    assert check_model, "This is only intended for CVAE"

    print('*' * 80)
    print("LOG-LIKELIHOOD LOWER BOUNDS ON TEST SUBSET")
    print('*' * 80)

    # TODO

def load_model_by_name(model, global_step, device=None):
    """
    Load model based on name and checkpoint iteration step
    :param model: Model object
    :param global_step: int
    :param device: string
    :return:
    """

    file_path = os.path.join('checkpoints',
                             model.name,
                             'model-{:05d}.pt'.format(global_step))
    state = torch.load(file_path, map_location=device)
    model.load_state_dict(state)
    print("loaded from {}".format(file_path))