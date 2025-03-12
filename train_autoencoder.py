from argparse import ArgumentParser
import os
import torch
from tqdm import tqdm
from autoencoder.ae_dataset import AE_dataset
from autoencoder.autoencoder import Autoencoder
from helpers import cos_loss, l2_loss_feat
from torch.utils.data import DataLoader


def train_ae(sequence, feature_extraction_model, args):
    ckpt_path = f"./data/{sequence}/{args.autoencoder_dir}"
    os.makedirs(ckpt_path, exist_ok=True)
    model = Autoencoder().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.num_epochs):
        print(f"Epoch: {epoch}")
        train_dataset = AE_dataset(epoch, sequence, feature_extraction_model, args.loader_step_size, args.features_dir)
        val_dataset = AE_dataset(epoch, sequence, feature_extraction_model, args.loader_step_size, args.features_dir,train=False)

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,
            pin_memory=False,
        )

        test_loader = DataLoader(
            dataset=val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False  , 
            pin_memory=False,
        )

        model.train()
        progress_bar = tqdm(enumerate(train_loader))
        for idx, feature in enumerate(train_loader):
            feature = feature[0].cuda()
            output = model(feature)
            l2loss = l2_loss_feat(output, feature) 
            cosloss = cos_loss(output, feature)
            loss = l2loss + cosloss * args.regularization_weight
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix({"Loss":"{:.8f}".format(loss)})
            progress_bar.update(1)
        progress_bar.close()

        eval_loss = 0.0
        model.eval()
        with torch.no_grad():
            for idx, feature in enumerate(test_loader):
                feature = feature[0].cuda()
                output = model(feature)
                loss = l2_loss_feat(output, feature) + cos_loss(output, feature)
                eval_loss += loss            
            eval_loss = eval_loss / len(val_dataset)
            print("eval_loss: {}".format(eval_loss))

        torch.save(model.state_dict(), os.path.join(ckpt_path, "best_ckpt.pth"))

if __name__ == "__main__":
    parser = ArgumentParser(description="Train args")
    parser.add_argument("-s","--sequence", type=str, required=True)
    parser.add_argument("-i","--feature_extraction_model", type=str, default="viclip")
    parser.add_argument("--autoencoder_dir", type=str, default="ae")
    parser.add_argument("--lr", type=float, default=0.0007)
    parser.add_argument("--regularization_weight", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--features_dir", type=str, default="interpolators")
    parser.add_argument("--loader_step_size", type=int, default=15)
    args = parser.parse_args()
    train_ae(args.sequence, args.feature_extraction_model, args)
