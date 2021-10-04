import torch
from random import sample


def train(model, optimizer, loss_graph_text,
          train_loader, val_loader, epochs,
          early_stopping, checkpoint_path, device='cpu'):
    '''
    :param model: the model to be trained
    :param optimizer: the optimizer for the training session
    :param loss_graph_text: the CL loss function for the representations of the graph and the text
    :param train_loader: the dataloader for the training data
    :param val_loader: the dataloader for the validation data
    :param epochs: the number of epochs for the training
    :param checkpoint_path: the path where the trained model will be saved
    :param device: 'cpu' or 'cuda'
    :return: the trained model (final version not the best one based on validation performance)
             and a dictionary with the losses (training and validation)
    '''

    # Initialize the minimum validation loss with a large value
    valid_loss_min = 1000000000
    acc_training_loss = []
    acc_val_loss = []

    # For applying early stopping
    valid_no_improvement = 0
    for epoch in range(epochs):
        training_loss = 0.0
        valid_loss = 0.0
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
            node_embeddings, adj_m, enc_sent, atte_mask, indexes_of_interest_shifted = batch

            # Run the forward pass for the batch
            loss_batch = 0
            for in1, in2, in3, in4, in5 in zip(node_embeddings, adj_m, enc_sent,
                                                   atte_mask, indexes_of_interest_shifted):
                # Place the inputs in the
                # selected device (GPU or CPU)
                in3 = in3.to(device)
                in4 = in4.to(device)

                output_graph, output_text, _ = model(x=in1,
                                                     adj=in2,
                                                     sent_id=in3,
                                                     mask=in4,
                                                     indexes_of_pairs=in5)

                loss_batch = loss_batch + loss_graph_text(output_graph, output_text)
                
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

        ###################
        # Validation step #
        ###################
        print('---------------')
        print('Validation Step')
        print('---------------')

        model.eval()
        c2 = 0
        with torch.no_grad():
            for batch in val_loader:
                # Take the inputs from the batch
                node_embeddings, adj_m, enc_sent, atte_mask, indexes_of_interest_shifted = batch

                for in1, in2, in3, in4, in5 in zip(node_embeddings, adj_m, enc_sent,
                                                   atte_mask, indexes_of_interest_shifted):
                    # Place the inputs in the
                    # selected device (GPU or CPU)
                    in3 = in3.to(device)
                    in4 = in4.to(device)

                    output_graph, output_text, _ = model(x=in1,
                                                         adj=in2,
                                                         sent_id=in3,
                                                         mask=in4,
                                                         indexes_of_pairs=in5)

                    valid_loss = valid_loss + loss_graph_text(output_graph, output_text).data.item()

                # Increment the counter (number of batches)
                c2 += 1

                if c2 % 30 == 0:
                    print('{} batches completed.'.format(c2))

            # Find the average validation loss over the batches
            valid_loss /= c2

            # Save the validation loss
            acc_val_loss.append(valid_loss)

        # Print the loss every 10 epochs
        # if epoch % 9 == 0:
        print('Epoch: {}'.format(epoch + 1))
        print('Training Loss: {:.4f}'.format(training_loss))
        print('Validation Loss: {:.4f}'.format(valid_loss))
        print('_________________________')

        # Check if the validation loss has been reduced in order to update the "best" checkpoint
        if valid_loss <= valid_loss_min:
            # Update the early stopping counter
            valid_no_improvement = 0

            print('Validation loss decreased ({:.4f} --> {:.4f}).  Saving model ...'.format(valid_loss_min, valid_loss))
            print('#############')
            print('#############')

            # Create checkpoint variable and add important data
            checkpoint = {'epoch': epoch + 1,
                          'valid_loss_min': valid_loss,
                          'state_dict': model.state_dict(),
                          'optimizer': optimizer.state_dict()
                          }

            # save checkpoint as best model
            torch.save(checkpoint, checkpoint_path + 'best_val_trained_model.pt')

            # Update the minimum validation loss
            valid_loss_min = valid_loss
        else:
            # No validation improvement
            valid_no_improvement += 1

        # Save the checkpoint at the end of each epoch
        # Create checkpoint variable and add important data
        checkpoint = {'epoch': epoch + 1,
                      'state_dict': model.state_dict(),
                      'optimizer': optimizer.state_dict(),
                      'training loss': training_loss,
                      'validation loss': valid_loss}

        torch.save(checkpoint, checkpoint_path + 'trained_model_epoch_' + str(epoch + 1) + '.pt')

        # Check for early stopping
        if valid_no_improvement >= early_stopping:
            print('Early stopping after {} epochs.'.format(str(epoch + 1)))
            break

    # Dictionary with the losses
    losses_dict = {'training loss': acc_training_loss,
                   'validation loss': acc_val_loss}

    return model, losses_dict

