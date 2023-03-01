import enum

from torch import nn as nn

import independent_batch_acquisition
from acquisition_batch import AcquisitionBatch
from acquisition_functions import AcquisitionFunction


class AcquisitionMethod(enum.Enum):
    independent = "independent"

    def acquire_batch(
        self,
        bayesian_model: nn.Module,
        acquisition_function: AcquisitionFunction,
        available_loader,
        num_classes,
        k,
        b,
        min_candidates_per_acquired_item,
        min_remaining_percentage,
        initial_percentage,
        reduce_percentage,
        device=None,
        train_loader=None,
        test_loader=None,
        target_loader=None,
        target_loader2=None,
    ) -> AcquisitionBatch:
        target_size = max(
            min_candidates_per_acquired_item * b, len(available_loader.dataset) * min_remaining_percentage // 100
        )

        if self == self.independent:
            return independent_batch_acquisition.compute_acquisition_bag(
                bayesian_model=bayesian_model,
                acquisition_function=acquisition_function,
                num_classes=num_classes,
                k=k,
                b=b,
                initial_percentage=initial_percentage,
                reduce_percentage=reduce_percentage,
                available_loader=available_loader,
                device=device,
                train_loader=train_loader,
                test_loader=test_loader,
                target_loader=None,
                target_loader2=None,
            )
        else:
            raise NotImplementedError(f"Unknown acquisition method {self}!")
