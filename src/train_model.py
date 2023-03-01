import ignite
from torch import optim as optim
from torch.nn import functional as F
from torch import nn

import ignite_restoring_score_guard
from ignite_progress_bar import ignite_progress_bar
from ignite_utils import epoch_chain, chain, log_epoch_results, store_epoch_results, store_iteration_results
from sampler_model import SamplerModel, NoDropoutModel
from typing import NamedTuple

import sys
import torch
from torch.autograd import Variable

class TrainModelResult(NamedTuple):
    num_epochs: int
    test_metrics: dict
    target_metrics: dict
    target_metrics2: dict



def build_metrics(add_confusion_matrix = False, num_classes=3):
    return {"accuracy": ignite.metrics.Accuracy(), "nll": ignite.metrics.Loss(F.nll_loss)}


def train_model(
    model: nn.Module,
    optimizer: optim.Optimizer,
    max_epochs,
    early_stopping_patience,
    num_inference_samples,
    test_loader,
    train_loader,
    validation_loader,
    log_interval,
    desc,
    device,
    dataset_name,
    lr_scheduler: optim.lr_scheduler._LRScheduler = None,
    num_lr_epochs=0,
    epoch_results_store=None,
    save_model=False,
    experiment_task_id="default_exp",
    target_loader=None,
    target_loader2=None,
    original_train=False,
    model_reuse=False,
) -> TrainModelResult:
    test_sampler = SamplerModel(model, k=100).to(device)
    validation_sampler = NoDropoutModel(model).to(device)
    training_sampler = SamplerModel(model, k=1).to(device)

    my_loss = F.nll_loss

    # trainer = ignite.engine.create_supervised_trainer(training_sampler, optimizer, F.nll_loss, device=device)
    trainer = ignite.engine.create_supervised_trainer(training_sampler, optimizer, my_loss, device=device)
    validation_evaluator = ignite.engine.create_supervised_evaluator(
        validation_sampler, metrics=build_metrics(), device=device
    )

    def out_of_patience():
        nonlocal num_lr_epochs
        if num_lr_epochs <= 0 or lr_scheduler is None:
            trainer.terminate()
        else:
            lr_scheduler.step()
            restoring_score_guard.patience = int(restoring_score_guard.patience * 1.5 + 0.5)
            print(f"New LRs: {[group['lr'] for group in optimizer.param_groups]}")
            num_lr_epochs -= 1

    if lr_scheduler is not None:
        print(f"LRs: {[group['lr'] for group in optimizer.param_groups]}")

    restoring_score_guard = ignite_restoring_score_guard.RestoringScoreGuard(
        patience=early_stopping_patience,
        score_function=lambda engine: engine.state.metrics["accuracy"],
        out_of_patience_callback=out_of_patience,
        module=model,
        optimizer=optimizer,
        training_engine=trainer,
        validation_engine=validation_evaluator,
    )

    if test_loader is not None:

        metrics = build_metrics()
        test_evaluator = ignite.engine.create_supervised_evaluator(test_sampler, metrics=metrics, device=device)
        ignite_progress_bar(test_evaluator, desc("Test Eval"), log_interval)
        chain(trainer, test_evaluator, test_loader)
        log_epoch_results(test_evaluator, "Test", trainer)

    ignite_progress_bar(trainer, desc("Training"), log_interval)
    ignite_progress_bar(validation_evaluator, desc("Validation Eval"), log_interval)

    # NOTE(blackhc): don't run a full test eval after every epoch.
    epoch_chain(trainer, validation_evaluator, validation_loader)
    log_epoch_results(validation_evaluator, "Validation", trainer)

    if epoch_results_store is not None:
        epoch_results_store["validations"] = []
        epoch_results_store["losses"] = []
        store_epoch_results(validation_evaluator, epoch_results_store["validations"])
        store_iteration_results(trainer, epoch_results_store["losses"], log_interval=2)

        if test_loader is not None:
            store_epoch_results(test_evaluator, epoch_results_store, name="test")

    if len(train_loader.dataset) > 0:
        print('train starts from here.')
        print(len(train_loader.dataset))

        if original_train:
            trainer.run(train_loader, max_epochs)
            train_acc = torch.tensor(0.0).cuda()
        else:
            best_model = model.state_dict()
            best_acc = 0
            train_best_acc = 0

            net = training_sampler
            optimizer = optim.Adam(net.parameters())


            for epoch in range(max_epochs):
                net = training_sampler
                net.train()
                net.training = True
                train_loss = 0
                correct = 0
                total = 0
                train_acc = 0.0
                train_size = 0
                for batch_idx, (inputs, targets) in enumerate(train_loader):
                    # if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda() # GPU settings
                    
                    optimizer.zero_grad()
                    inputs, targets = Variable(inputs), Variable(targets)
                    outputs = net(inputs) 
                    loss = my_loss(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += predicted.eq(targets.data).cpu().sum()
                    
                    train_size += total
                    train_acc += correct

                    sys.stdout.write('\r')
                    sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
                            %(epoch+1, max_epochs, batch_idx+1, loss.item(), 100.*correct/total))
                    sys.stdout.flush()

                train_acc /= train_size

                net.bayesian_net.set_feature_autograd(True)

            trainer.run(train_loader, 0)

    else:
        test_evaluator.run(test_loader)

    num_epochs = max_epochs

    test_metrics = None
    if test_loader is not None:
        test_metrics = test_evaluator.state.metrics
        test_metrics['train_accuracy'] = train_acc.cpu().detach().numpy().item()
 
    return TrainModelResult(num_epochs, test_metrics, {}, {})