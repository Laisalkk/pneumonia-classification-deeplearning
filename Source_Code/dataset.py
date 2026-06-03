# ============================================================
# DATASET
# ============================================================

from PIL import Image

from torch.utils.data import Dataset

# ============================================================
# PNEUMONIA DATASET
# ============================================================

class PneumoniaDataset(Dataset):

    def __init__(
        self,
        dataframe,
        transform=None
    ):
        """
        Parameters
        ----------
        dataframe : pandas.DataFrame

            Wajib memiliki:
            - image_path
            - label

        transform : torchvision.transforms
        """

        self.dataframe = dataframe.reset_index(
            drop=True
        )

        self.transform = transform

    def __len__(self):

        return len(self.dataframe)

    def __getitem__(self, idx):

        row = self.dataframe.iloc[idx]

        image_path = row["image_path"]

        label = int(
            row["label"]
        )

        try:

            image = Image.open(
                image_path
            ).convert("L")

        except Exception as e:

            raise RuntimeError(
                f"Failed loading image:\n"
                f"{image_path}\n"
                f"{e}"
            )

        if self.transform:

            image = self.transform(
                image
            )

        return image, label