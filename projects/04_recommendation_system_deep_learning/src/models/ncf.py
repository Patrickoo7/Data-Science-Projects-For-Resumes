"""
Neural Collaborative Filtering (NCF) implementation.
Combines GMF (Generalized Matrix Factorization) and MLP.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GMF(nn.Module):
    """Generalized Matrix Factorization model."""

    def __init__(self, num_users, num_items, embedding_dim=64):
        super(GMF, self).__init__()

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.output_layer = nn.Linear(embedding_dim, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)

        # Element-wise product
        gmf_vector = user_emb * item_emb

        output = self.output_layer(gmf_vector)
        return output.squeeze()


class MLP(nn.Module):
    """Multi-Layer Perceptron for collaborative filtering."""

    def __init__(
        self,
        num_users,
        num_items,
        embedding_dim=64,
        layers=[128, 64, 32],
        dropout=0.2
    ):
        super(MLP, self).__init__()

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # MLP layers
        mlp_modules = []
        input_size = embedding_dim * 2

        for layer_size in layers:
            mlp_modules.append(nn.Linear(input_size, layer_size))
            mlp_modules.append(nn.ReLU())
            mlp_modules.append(nn.Dropout(dropout))
            input_size = layer_size

        self.mlp_layers = nn.Sequential(*mlp_modules)
        self.output_layer = nn.Linear(layers[-1], 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

        for m in self.mlp_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)

        # Concatenate embeddings
        mlp_vector = torch.cat([user_emb, item_emb], dim=-1)

        # Pass through MLP
        mlp_output = self.mlp_layers(mlp_vector)
        output = self.output_layer(mlp_output)

        return output.squeeze()


class NeuMF(nn.Module):
    """
    Neural Matrix Factorization.
    Combines GMF and MLP for improved performance.
    """

    def __init__(
        self,
        num_users,
        num_items,
        gmf_embedding_dim=64,
        mlp_embedding_dim=64,
        mlp_layers=[128, 64, 32],
        dropout=0.2
    ):
        super(NeuMF, self).__init__()

        # GMF part
        self.gmf_user_embedding = nn.Embedding(num_users, gmf_embedding_dim)
        self.gmf_item_embedding = nn.Embedding(num_items, gmf_embedding_dim)

        # MLP part
        self.mlp_user_embedding = nn.Embedding(num_users, mlp_embedding_dim)
        self.mlp_item_embedding = nn.Embedding(num_items, mlp_embedding_dim)

        # MLP layers
        mlp_modules = []
        input_size = mlp_embedding_dim * 2

        for layer_size in mlp_layers:
            mlp_modules.append(nn.Linear(input_size, layer_size))
            mlp_modules.append(nn.ReLU())
            mlp_modules.append(nn.Dropout(dropout))
            input_size = layer_size

        self.mlp_layers = nn.Sequential(*mlp_modules)

        # Final prediction layer
        self.output_layer = nn.Linear(gmf_embedding_dim + mlp_layers[-1], 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.gmf_user_embedding.weight, std=0.01)
        nn.init.normal_(self.gmf_item_embedding.weight, std=0.01)
        nn.init.normal_(self.mlp_user_embedding.weight, std=0.01)
        nn.init.normal_(self.mlp_item_embedding.weight, std=0.01)

    def forward(self, user_ids, item_ids):
        # GMF part
        gmf_user_emb = self.gmf_user_embedding(user_ids)
        gmf_item_emb = self.gmf_item_embedding(item_ids)
        gmf_vector = gmf_user_emb * gmf_item_emb

        # MLP part
        mlp_user_emb = self.mlp_user_embedding(user_ids)
        mlp_item_emb = self.mlp_item_embedding(item_ids)
        mlp_vector = torch.cat([mlp_user_emb, mlp_item_emb], dim=-1)
        mlp_vector = self.mlp_layers(mlp_vector)

        # Concatenate GMF and MLP
        neumf_vector = torch.cat([gmf_vector, mlp_vector], dim=-1)

        # Final prediction
        output = self.output_layer(neumf_vector)
        return output.squeeze()


class DeepCrossNetwork(nn.Module):
    """
    Deep & Cross Network for feature interactions.
    """

    def __init__(
        self,
        num_users,
        num_items,
        embedding_dim=64,
        cross_layers=3,
        deep_layers=[128, 64, 32],
        dropout=0.2
    ):
        super(DeepCrossNetwork, self).__init__()

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        input_dim = embedding_dim * 2

        # Cross Network
        self.cross_layers = nn.ModuleList([
            nn.Linear(input_dim, input_dim) for _ in range(cross_layers)
        ])

        # Deep Network
        deep_modules = []
        deep_input = input_dim

        for layer_size in deep_layers:
            deep_modules.append(nn.Linear(deep_input, layer_size))
            deep_modules.append(nn.BatchNorm1d(layer_size))
            deep_modules.append(nn.ReLU())
            deep_modules.append(nn.Dropout(dropout))
            deep_input = layer_size

        self.deep_network = nn.Sequential(*deep_modules)

        # Output layer
        self.output_layer = nn.Linear(input_dim + deep_layers[-1], 1)

    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)

        x = torch.cat([user_emb, item_emb], dim=-1)
        x0 = x

        # Cross Network
        cross_out = x
        for cross_layer in self.cross_layers:
            cross_out = x0 * cross_layer(cross_out) + cross_out

        # Deep Network
        deep_out = self.deep_network(x)

        # Combine and predict
        combined = torch.cat([cross_out, deep_out], dim=-1)
        output = self.output_layer(combined)

        return output.squeeze()


class AutoEncoder(nn.Module):
    """
    Autoencoder for collaborative filtering.
    """

    def __init__(
        self,
        num_items,
        encoder_dims=[512, 256, 128],
        bottleneck_dim=64,
        dropout=0.3
    ):
        super(AutoEncoder, self).__init__()

        # Encoder
        encoder_modules = []
        input_dim = num_items

        for dim in encoder_dims:
            encoder_modules.append(nn.Linear(input_dim, dim))
            encoder_modules.append(nn.ReLU())
            encoder_modules.append(nn.Dropout(dropout))
            input_dim = dim

        encoder_modules.append(nn.Linear(input_dim, bottleneck_dim))
        self.encoder = nn.Sequential(*encoder_modules)

        # Decoder
        decoder_modules = []
        input_dim = bottleneck_dim

        for dim in reversed(encoder_dims):
            decoder_modules.append(nn.Linear(input_dim, dim))
            decoder_modules.append(nn.ReLU())
            decoder_modules.append(nn.Dropout(dropout))
            input_dim = dim

        decoder_modules.append(nn.Linear(input_dim, num_items))
        self.decoder = nn.Sequential(*decoder_modules)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def get_user_embedding(self, x):
        return self.encoder(x)


if __name__ == "__main__":
    # Test models
    num_users = 1000
    num_items = 500
    batch_size = 32

    user_ids = torch.randint(0, num_users, (batch_size,))
    item_ids = torch.randint(0, num_items, (batch_size,))

    print("Testing NeuMF...")
    neumf = NeuMF(num_users, num_items)
    output = neumf(user_ids, item_ids)
    print(f"Output shape: {output.shape}")

    print("\nTesting Deep & Cross Network...")
    dcn = DeepCrossNetwork(num_users, num_items)
    output = dcn(user_ids, item_ids)
    print(f"Output shape: {output.shape}")

    print("\nAll models tested successfully!")
