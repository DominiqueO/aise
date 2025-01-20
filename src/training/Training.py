
import torch

def train_NN(model, optimizer, scheduler, loss, training_set, testing_set, epochs=50, freq_print=1, save=False,
             save_name='default', squeeze_dim=2):
    loss_history = []
    l2_history = []
    for epoch in range(epochs):
        train_mse = 0.0
        for step, (input_batch, output_batch) in enumerate(training_set):
            optimizer.zero_grad()
            output_pred_batch = model(input_batch).squeeze(squeeze_dim)
            loss_f = loss(output_pred_batch, output_batch)
            loss_f.backward()
            optimizer.step()
            train_mse += loss_f.item()
        train_mse /= len(training_set)

        scheduler.step()

        with torch.no_grad():
            model.eval()
            test_relative_l2 = 0.0
            for step, (input_batch, output_batch) in enumerate(testing_set):
                output_pred_batch = model(input_batch).squeeze(squeeze_dim)
                loss_f = (torch.mean((output_pred_batch - output_batch) ** 2) / torch.mean(
                    output_batch ** 2)) ** 0.5 * 100
                test_relative_l2 += loss_f.item()
            test_relative_l2 /= len(testing_set)

        if epoch % freq_print == 0: print("######### Epoch:", epoch, " ######### Train Loss:", train_mse,
                                          " ######### Relative L2 Test Norm:", test_relative_l2)
        loss_history.append(train_mse)
        l2_history.append(test_relative_l2)

    if save:
        torch.save(model.state_dict(), save_name + ".pt")
    return model, loss_history, l2_history