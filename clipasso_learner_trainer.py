import argparse
import utils.svgutils
import torch
import matplotlib.pyplot as plt
import utils.svgutils
import wandb
import datetime
import os
from models.mlp import MLP
from datasets.clipasso import ClipassoDataset
from torch.utils.data import DataLoader

# set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def visualize_examples(target: torch.Tensor, 
                       output: torch.Tensor, 
                       epoch: int,
                       output_path: str,
                       test: bool=False,
                       use_wandb: bool=False):
    # show image and output pair
    gt_img = 1.0 - target.repeat(1, 3, 1, 1).squeeze().permute(1, 2, 0).detach().cpu().numpy()
    out_path = output.cpu().detach()
    out_path = utils.svgutils.decanonizePaths(out_path)
    out_svgdict = utils.svgutils.tensor2SVG(out_path)
    out_img = utils.svgutils.renderCLipassoSVG(out_svgdict["shapes"], out_svgdict["shape_groups"])
    # plot the images side by side and save them
    _, ax = plt.subplots(1, 2)
    ax[0].imshow(gt_img)
    ax[0].set_title("Target GT")
    ax[1].imshow(out_img.squeeze().detach().cpu().numpy())
    ax[1].set_title("Output")
    # set the title
    plt.suptitle(f"Epoch {epoch} {'Test' if test else 'Train'}")
    plt.savefig(os.path.join(output_path, f"epoch_{epoch}_{'test' if test else 'train'}.png"))
    if use_wandb:
        wandb.log({f"{'test' if test else 'train'}_example": wandb.Image(plt)}, step=epoch)
    plt.close()

def train(args: dict):
    # Load the dataset
    train_dataset = ClipassoDataset("/home/etaisella/data/256_ObjectCategories_clipasso/train")
    #train_dataset = ClipassoDataset("/home/etaisella/data/clipasso_test/train")
    test_dataset = ClipassoDataset("/home/etaisella/data/256_ObjectCategories_clipasso/test")
    #test_dataset = ClipassoDataset("/home/etaisella/data/clipasso_test/test")
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=False)
    print(f"Length of CLipasso Train Dataset: {len(train_dataset)}")
    print(f"Length of CLipasso Test Dataset: {len(test_dataset)}")

    # Create the MLP model
    model = MLP().to(device)

    # Define the optimizer and loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.MSELoss()

    # set arrays for losses
    train_losses = []
    test_losses = []

    # Training loop
    for epoch in range(args.epoch_count):
        print(f"Epoch [{epoch+1}/{args.epoch_count}]")
        total_loss_train = 0
        for iter, (images, paths) in enumerate(train_dataloader):
            # canonize the paths
            cpaths = utils.svgutils.canonizePaths(paths).to(device)

            # Perform forward pass
            outputs = model(images.to(device))

            # Compute loss
            loss = criterion(outputs, cpaths)
            total_loss_train += loss.item()

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print training progress
        # normalize the total loss by step
        total_loss_train /= (iter + 1)
        print(f"Epoch [{epoch+1}/{args.epoch_count}], Loss: {total_loss_train}")
        train_losses.append(total_loss_train)
        if ((epoch + 1) % args.test_frequency == 0) or (epoch == 0):
            visualize_examples(images[0], outputs[0], epoch, args.output_path, test=False, use_wandb=args.use_wandb)
            # Test the model
            model.eval()
            with torch.no_grad():
                total_loss_test = 0
                for iter, (t_images, t_paths) in enumerate(test_dataloader):
                    # canonize the paths
                    t_cpaths = utils.svgutils.canonizePaths(t_paths).to(device)

                    # Perform forward pass
                    t_outputs = model(t_images.to(device))

                    # Compute loss
                    loss = criterion(t_outputs, t_cpaths)
                    total_loss_test += loss.item()

                # Print test progress
                # normalize the total loss by step
                total_loss_test /= (iter + 1)
                print(f"Epoch [{epoch+1}/{args.epoch_count}], Test Loss: {total_loss_test}")
                # log loss and visualize examples
                visualize_examples(t_images[0], t_outputs[0], epoch, args.output_path, test=True, use_wandb=args.use_wandb)
                test_losses.append(total_loss_test)
                # plot test and train losses
                plt.plot(train_losses, label="Train Loss")
                plt.plot(test_losses, label="Test Loss")
                plt.legend()
                plt.savefig(os.path.join(args.output_path, f"losses_{epoch}.png"))
                plt.close()
                if args.use_wandb:
                    wandb.log({"train_loss": total_loss_train, "test_loss": total_loss_test}, step=epoch)
            model.train()
           
    # Save the trained model
    torch.save(model.state_dict(), "trained_model.pth")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_path", type=str, default="output_folder", help="Path to save output images and models")
    parser.add_argument("--learning_rate", type=float, default=0.006, help="Learning rate for training")
    parser.add_argument("--epoch_count", type=int, default=10000, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--test_frequency", type=int, default=20, help="Frequency of testing during training")
    parser.add_argument("--use_wandb", type=bool, default=True, help="Wether or not to use wandb") # TODO (ES): Change to False before release
    args = parser.parse_args()

    # make output folder if it does not exist
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # initialize wandb (if using)
    # TODO (ES): Delete my username before release
    if args.use_wandb:
        wandb.init(project='SVGLearner', entity='etaisella', name="clipasso training",
                   id=wandb.util.generate_id())
        wandb.config.update(args)

    # Start training
    train(args)