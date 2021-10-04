import torch
from random import sample


def train(model, optimizer, loss,
          train_loader, epochs, checkpoint_path, device='cpu'):
    '''
    :param model: the model to be trained
    :param optimizer: the optimizer for the training session
    :param loss: the CL loss function for the NE representations
    :param train_loader: the dataloader for the training data
    :param epochs: the number of epochs for the training
    :param checkpoint_path: the path where the trained model will be saved
    :param device: 'cpu' or 'cuda'
    :return: the trained model (final version not the best one based on validation performance)
             and a dictionary with the losses (training and validation)
    '''
    acc_training_loss = []
    for epoch in range(epochs):
        training_loss = 0.0
        #################
        # Training step #
        #################
        print('-------------')
        print('Training Step')
        print('-------------')
        model.train()
        c1 = 0
        for batch in train_loader:

            # Take the inputs from the batch
            enc_sent, atte_mask, ne_tags = batch

            # Run the forward pass for the batch
            ne_rep_list = []
            ne_tag_list = []
            for in1, in2, in3 in zip(enc_sent, atte_mask, ne_tags):
                # Place the inputs in the
                # selected device (GPU or CPU)
                in1 = in1.to(device)
                in2 = in2.to(device)

                ne_rep = model(sent_id=in1, mask=in2)

                ne_rep_list.extend(ne_rep)
                ne_tag_list.extend(in3)


            # Find the ne loss in the batch-level
            loss_batch = loss(ne_rep_list, ne_tag_list)

            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the variables it will update (which are the learnable
            # weights of the model). This is because by default, gradients are
            # accumulated in buffers( i.e, not overwritten) whenever .backward() is called.
            optimizer.zero_grad()

            # Run a backpropagation pass
            loss_batch.backward()

            # Gradient descent step
            optimizer.step()

            # Add the loss of the batch
            training_loss += loss_batch.data.item()

            # Increment the counter (number of batches)
            c1 += 1

            if c1 % 30 == 0:
                print('{} batches completed.'.format(c1))

        # Find the average training loss over the batches
        training_loss /= c1

        # Save the training loss
        acc_training_loss.append(training_loss)

        # Print the loss every 10 epochs
        # if epoch % 9 == 0:
        print('Epoch: {}'.format(epoch + 1))
        print('Training Loss: {:.4f}'.format(training_loss))
        print('_________________________')

        # Save the checkpoint at the end of each epoch
        # Create checkpoint variable and add important data
        if epoch + 1 == epochs:
            # Create checkpoint variable and add important data
            checkpoint = {'epoch': epoch + 1,
                          'state_dict': model.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'training loss': training_loss}

            # save checkpoint as best model
            torch.save(checkpoint, checkpoint_path + 'final_trained_model.pt')
        '''
        else:
            checkpoint = {'epoch': epoch + 1,
                          'state_dict': model.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'training loss': training_loss}

            torch.save(checkpoint, checkpoint_path + 'trained_model_epoch_' + str(epoch + 1) + '.pt')
        '''
    # Dictionary with the losses
    losses_dict = {'training loss': acc_training_loss}

    return model, losses_dict