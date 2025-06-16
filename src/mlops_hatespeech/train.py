from mlops_hatespeech.data import MyDataset
from mlops_hatespeech.model import Model


def train():
    dataset = MyDataset("data/raw")
    model = Model()
    # add rest of your training code here

if __name__ == "__main__":
    train()
