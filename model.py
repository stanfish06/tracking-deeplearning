import os, torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import lightning as L


class EncodeDecode(nn.Module):
    def __init__(self, img_size):
        super(EncodeDecode, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        conv_out_size = ((img_size - 2 - 1) + 1) - 2 - 1 + 1
        self.lstm = nn.LSTM(10, 10, 1, batch_first=True)
        self.fc1 = nn.Linear(conv_out_size * conv_out_size * 64, 128)
        self.fc2 = nn.Linear(128, 10)
        # subnetwork to embedding spatial distance
        self.distance_fc1 = torch.nn.Linear(
            1, 64
        )  # 1 input (distance), 64 hidden units
        self.distance_fc2 = torch.nn.Linear(64, 1)  # Output 1 weight per neighbor

    # here batch_size is the number of cells
    # seq_len is the number of frames for each cell, including the current time frame
    # channels is the number of channels in the image, I would assume it to be 1 for now
    # img tp1 will contain k neighboring cells' images at the next time frame, I will handle this in data loader
    def forward(self, img_seq_t, img_tp1, dist_to_neighbors):
        batch_size, seq_len, channels, height, width = img_seq_t.size()
        _, k, _, _, _ = img_tp1.size()
        # process img_seq_t
        img_seq_t = img_seq_t.view(batch_size * seq_len, channels, height, width)
        x = self.conv1(img_seq_t)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)

        # flatten the tensor
        x = x.view(batch_size * seq_len, -1)
        # linear decode
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = x.view(batch_size, seq_len, -1)

        # apply lstm
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # get the last time frame

        # process img_tp1
        img_tp1 = img_tp1.view(batch_size * k, channels, height, width)
        y = self.conv1(img_tp1)
        y = F.relu(y)
        y = self.conv2(y)
        y = F.relu(y)
        y = y.view(batch_size * k, -1)
        # linear decode
        y = self.fc1(y)
        y = F.relu(y)
        y = self.fc2(y)
        y = F.relu(y)
        y = y.view(batch_size, k, -1)

        dist_to_neighbors = dist_to_neighbors.unsqueeze(-1)
        dist_to_neighbors = dist_to_neighbors.view(batch_size * k, -1)
        # Pass the distances through the sub-network to learn the distance weights
        dist_weights = self.distance_fc1(dist_to_neighbors)
        dist_weights = F.relu(dist_weights)
        dist_weights = self.distance_fc2(dist_weights)
        dist_weights = F.relu(dist_weights)
        dist_weights = dist_weights.view(batch_size, 1, k)

        # try cosine similarity
        lstm_out_norm = F.normalize(lstm_out, p=2, dim=-1)
        y_norm = F.normalize(y, p=2, dim=-1)
        similarity_matrix = torch.bmm(
            lstm_out_norm.unsqueeze(1), y_norm.transpose(1, 2)
        )
        weighted_similarity_matrix = similarity_matrix * dist_weights
        probability_matrix = F.softmax(weighted_similarity_matrix, dim=-1)
        return probability_matrix


class EncodeDecodeModel(L.LightningModule):
    def __init__(self, img_size):
        super(EncodeDecodeModel, self).__init__()
        self.model = EncodeDecode(img_size)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, img_seq_t, img_tp1, dist_to_neighbors):
        return self.model(img_seq_t, img_tp1, dist_to_neighbors)

    def training_step(self, batch, batch_idx):
        img_seq_t, img_tp1, dist_to_neighbors, labels = batch
        probability_matrix = self.model(img_seq_t, img_tp1, dist_to_neighbors)
        prob_flat = probability_matrix.view(-1, probability_matrix.size(-1))
        labels_flat = labels.view(-1)
        loss = self.loss_fn(prob_flat, labels_flat)
        self.log("training loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        img_seq_t, img_tp1, dist_to_neighbors, labels = batch
        probability_matrix = self.model(img_seq_t, img_tp1, dist_to_neighbors)
        prob_flat = probability_matrix.view(-1, probability_matrix.size(-1))
        labels_flat = labels.view(-1)
        loss = self.loss_fn(prob_flat, labels_flat)
        self.log("test loss", loss, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        img_seq_t, img_tp1, dist_to_neighbors, labels = batch
        probability_matrix = self.model(img_seq_t, img_tp1, dist_to_neighbors)
        prob_flat = probability_matrix.view(-1, probability_matrix.size(-1))
        labels_flat = labels.view(-1)
        loss = self.loss_fn(prob_flat, labels_flat)
        self.log("validation loss", loss, prog_bar=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer


def main():
    img_size = 28  # Example image size
    model = EncodeDecodeModel(img_size)

    # Create dummy data
    batch_size = 4
    seq_len = 5
    channels = 1
    height = img_size
    width = img_size
    k = 3  # number of neighboring cells

    img_seq_t = torch.randn(batch_size, seq_len, channels, height, width)
    img_tp1 = torch.randn(batch_size, k, channels, height, width)
    dist_to_neighbors = torch.randn(batch_size, k)

    # Forward pass
    output = model(img_seq_t, img_tp1, dist_to_neighbors)
    print(output)


# if __name__ == "__main__":
#     main()
