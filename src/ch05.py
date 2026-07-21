# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown] id="h1ispDL1G4Fc"
# # 第5章

# %% id="a_gEnU8rWN3e" colab={"base_uri": "https://localhost:8080/"} outputId="ddeb6bd9-07ae-4db7-ca56-9c1d2d6a5603"
# !pip install pyro-ppl==1.8.4
# !pip install torchvision

# %% [markdown] id="Rsu5QxOW0aAx"
# ## リスト5.1

# %% id="lfWTRtPoWF5H"
import torch
USE_CUDA = False
DEVICE_TYPE = torch.device("cuda" if USE_CUDA else "cpu")

# %% [markdown] id="6UoHQdJpG4Fe"
# ## リスト5.2

# %% id="_GoP_qMKWF5I"
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
from torchvision import transforms

class CombinedDataset(Dataset):
    def __init__(self, csv_file):
        self.dataset = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        images = self.dataset.iloc[idx, 3:]
        images = np.array(images, dtype='float32')/255.
        images = images.reshape(28, 28)
        transform = transforms.ToTensor()
        images = transform(images)
        digits = self.dataset.iloc[idx, 2]
        digits = np.array([digits], dtype='int')
        is_handwritten = self.dataset.iloc[idx, 1]
        is_handwritten = np.array([is_handwritten], dtype='float32')
        return images, digits, is_handwritten

# %% [markdown] id="6UGXmWzuJlpL"
# ## リスト5.3

# %% id="S4jn_XvTJlxH"
from torch.utils.data import DataLoader
from torch.utils.data import random_split

def setup_dataloaders(batch_size=64, use_cuda=USE_CUDA):
    combined_dataset = CombinedDataset(
"https://raw.githubusercontent.com/altdeep/causalAI/master/datasets/combined_mnist_tmnist_data.csv"
    )
    n = len(combined_dataset)
    train_size = int(0.8 * n)
    test_size = n - train_size
    train_dataset, test_dataset = random_split(
        combined_dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    kwargs = {'num_workers': 1, 'pin_memory': use_cuda}
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **kwargs
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        **kwargs
    )
    return train_loader, test_loader

# %% [markdown] id="fRXUclDPpKpd"
# ## リスト5.4

# %% id="pQEwj1mh-TTj"
from torch import nn

class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super().__init__()
        img_dim = 28 * 28
        digit_dim = 10
        is_handwritten_dim = 1
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        encoding_dim = z_dim + digit_dim + is_handwritten_dim
        self.fc1 = nn.Linear(encoding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, img_dim)

    def forward(self, z, digit, is_handwritten):
        input = torch.cat([z, digit, is_handwritten], dim=1)
        hidden = self.softplus(self.fc1(input))
        img_param = self.sigmoid(self.fc2(hidden))
        return img_param

# %% [markdown] id="aYlhi-IlpshP"
# ## リスト5.5

# %% id="ndMbxPUfJlqu"
import pyro
import pyro.distributions as dist

dist.enable_validation(False)
def model(self, data_size=1):
    pyro.module("decoder", self.decoder)
    options = dict(dtype=torch.float32, device=DEVICE_TYPE)
    z_loc = torch.zeros(data_size, self.z_dim, **options)
    z_scale = torch.ones(data_size, self.z_dim, **options)
    z = pyro.sample("Z", dist.Normal(z_loc, z_scale).to_event(1))
    p_digit = torch.ones(data_size, 10, **options)/10
    digit = pyro.sample(
        "digit",
        dist.OneHotCategorical(p_digit)
    )
    p_is_handwritten = torch.ones(data_size, 1, **options)/2
    is_handwritten = pyro.sample(
        "is_handwritten",
        dist.Bernoulli(p_is_handwritten).to_event(1)
    )
    img_param = self.decoder(z, digit, is_handwritten)
    img = pyro.sample("img", dist.Bernoulli(img_param).to_event(1))
    return img, digit, is_handwritten

# %% [markdown] id="49xn77L8qCmn"
# ## リスト5.6

# %% id="RQGgpOPSWOKa"
def training_model(self, img, digit, is_handwritten, batch_size):
    conditioned_on_data = pyro.condition(
        self.model,
        data={
            "digit": digit,
            "is_handwritten": is_handwritten,
            "img": img
        }
    )
    with pyro.plate("data", batch_size):
        img, digit, is_handwritten = conditioned_on_data(batch_size)
    return img, digit, is_handwritten

# %% [markdown] id="ia-xBsT6qioA"
# ## リスト5.7

# %% id="UGGxsLWIWkvG"
class Encoder(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super().__init__()
        img_dim = 28 * 28
        digit_dim = 10
        is_handwritten_dim = 1
        self.softplus = nn.Softplus()
        input_dim = img_dim + digit_dim + is_handwritten_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)

    def forward(self, img, digit, is_handwritten):
        input = torch.cat([img, digit, is_handwritten], dim=1)
        hidden = self.softplus(self.fc1(input))
        z_loc = self.fc21(hidden)
        z_scale = torch.exp(self.fc22(hidden))
        return z_loc, z_scale

# %% [markdown] id="TP920mmcqtsS"
# ## リスト5.8

# %% id="xy2T3-CtZp5d"
def training_guide(self, img, digit, is_handwritten, batch_size):
    pyro.module("encoder", self.encoder)
    options = dict(dtype=torch.float32, device=DEVICE_TYPE)
    with pyro.plate("data", batch_size):
        z_loc, z_scale = self.encoder(img, digit, is_handwritten)
        normal_dist = dist.Normal(z_loc, z_scale).to_event(1)
        z = pyro.sample("Z", normal_dist)

# %% [markdown] id="Bbit-PBlrayq"
# ## リスト5.9

# %% id="o5vOr1GXe_3P"
class VAE(nn.Module):
    def __init__(
        self,
        z_dim=50,
        hidden_dim=400,
        use_cuda=USE_CUDA,
    ):
        super().__init__()
        self.use_cuda = use_cuda
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.setup_networks()

    def setup_networks(self):
        self.encoder = Encoder(self.z_dim, self.hidden_dim)
        self.decoder = Decoder(self.z_dim, self.hidden_dim)
        if self.use_cuda:
            self.cuda()

    model = model
    training_model = training_model
    training_guide = training_guide

# %% [markdown] id="e8_IhBKK0kpO"
# ## リスト5.10

# %% id="ohxiEjB30lHa"
def plot_image(img, title=None):
    fig = plt.figure()
    plt.imshow(img.cpu(), cmap='Greys_r', interpolation='nearest')
    if title is not None:
        plt.title(title)
    plt.show()

# %% [markdown] id="BTFLSyVW06tr"
# ## リスト5.11

# %% id="QqBkhFAy066G"
import matplotlib.pyplot as plt

def reconstruct_img(vae, img, digit, is_hw, use_cuda=USE_CUDA):
    img = img.reshape(-1, 28 * 28)
    digit = F.one_hot(torch.tensor(digit), 10)
    is_hw = torch.tensor(is_hw).unsqueeze(0)
    if use_cuda:
        img = img.cuda()
        digit = digit.cuda()
        is_hw = is_hw.cuda()
    z_loc, z_scale = vae.encoder(img, digit, is_hw)
    z = dist.Normal(z_loc, z_scale).sample()
    img_expectation = vae.decoder(z, digit, is_hw)
    return img_expectation.squeeze().view(28, 28).detach()

def compare_images(img1, img2):
    fig = plt.figure()
    ax0 = fig.add_subplot(121)
    plt.imshow(img1.cpu(), cmap='Greys_r', interpolation='nearest')
    plt.axis('off')
    plt.title('original')
    ax1 = fig.add_subplot(122)
    plt.imshow(img2.cpu(), cmap='Greys_r', interpolation='nearest')
    plt.axis('off')
    plt.title('reconstruction')
    plt.show()

# %% [markdown] id="79MHVV7313mT"
# ## リスト5.12

# %% id="fmSgUQrl1tFN"
import torch.nn.functional as F

def get_random_example(loader):
    random_idx = np.random.randint(0, len(loader.dataset))
    img, digit, is_handwritten = loader.dataset[random_idx]
    return img.squeeze(), digit, is_handwritten

def reshape_data(img, digit, is_handwritten):
    digit = F.one_hot(digit, 10).squeeze()
    img = img.reshape(-1, 28*28)
    return img, digit, is_handwritten

def generate_coded_data(vae, use_cuda=USE_CUDA):
    z_loc = torch.zeros(1, vae.z_dim)
    z_scale = torch.ones(1, vae.z_dim)
    z = dist.Normal(z_loc, z_scale).to_event(1).sample()
    p_digit = torch.ones(1, 10)/10
    digit = dist.OneHotCategorical(p_digit).sample()
    p_is_handwritten = torch.ones(1, 1)/2
    is_handwritten = dist.Bernoulli(p_is_handwritten).sample()
    if use_cuda:
        z = z.cuda()
        digit = digit.cuda()
        is_handwritten = is_handwritten.cuda()
    img = vae.decoder(z, digit, is_handwritten)
    return img, digit, is_handwritten

def generate_data(vae, use_cuda=USE_CUDA):
    img, digit, is_handwritten = generate_coded_data(vae, use_cuda)
    img = img.squeeze().view(28, 28).detach()
    digit = torch.argmax(digit, 1)
    is_handwritten = torch.argmax(is_handwritten, 1)
    return img, digit, is_handwritten

# %% [markdown] id="wFp3sIBu2-eW"
# ## リスト5.13

# %% id="iEbyt3uo3Fhm"
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

pyro.clear_param_store()
vae = VAE()
train_loader, test_loader = setup_dataloaders(batch_size=256)
svi_adam = Adam({"lr": 1.0e-3})
model = vae.training_model
guide = vae.training_guide
svi = SVI(model, guide, svi_adam, loss=Trace_ELBO())

# %% [markdown] id="klKFXtLpTOos"
# ## リスト5.14

# %% id="XSviOogCTN2d"
def test_epoch(vae, test_loader):
    epoch_loss_test = 0
    for img, digit, is_hw in test_loader:
        batch_size = img.shape[0]
        if USE_CUDA:
            img = img.cuda()
            digit = digit.cuda()
            is_hw = is_hw.cuda()
        img, digit, is_hw = reshape_data(
            img, digit, is_hw
        )
        epoch_loss_test += svi.evaluate_loss(
            img, digit, is_hw, batch_size
        )
    test_size = len(test_loader.dataset)
    avg_loss = epoch_loss_test/test_size
    print("Epoch: {} avg. test loss: {}".format(epoch, avg_loss))
    print("Comparing a random test image to its reconstruction:")
    random_example = get_random_example(test_loader)
    img_r, digit_r, is_hw_r = random_example
    img_recon = reconstruct_img(vae, img_r, digit_r, is_hw_r)
    compare_images(img_r, img_recon)
    print("Generate a random image from the model:")
    img_gen, digit_gen, is_hw_gen = generate_data(vae)
    plot_image(img_gen, "Generated Image")
    print("Intended digit: ", int(digit_gen))
    print("Intended as handwritten: ", bool(is_hw_gen == 1))

# %% [markdown] id="pmvAUfzZ3X4M"
# ## リスト5.15

# %% id="krc0vcPv3Zqx" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="27747b85-41e6-4a95-abbe-287b9fcfdf59"
NUM_EPOCHS = 2500
TEST_FREQUENCY = 10

train_loss = []
train_size = len(train_loader.dataset)

for epoch in range(0, NUM_EPOCHS+1):
    loss = 0
    for img, digit, is_handwritten in train_loader:
        batch_size = img.shape[0]
        if USE_CUDA:
            img = img.cuda()
            digit = digit.cuda()
            is_handwritten = is_handwritten.cuda()
        img, digit, is_handwritten = reshape_data(
            img, digit, is_handwritten
        )
        loss += svi.step(
            img, digit, is_handwritten, batch_size
        )
    avg_loss = loss / train_size
    print("Epoch: {} avgs training loss: {}".format(epoch, loss))
    train_loss.append(avg_loss)
    if epoch % TEST_FREQUENCY == 0:
        test_epoch(vae, test_loader)
