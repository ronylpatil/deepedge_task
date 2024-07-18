import numpy as np


def inverse_dataset(
    train_size: int, test_size: int, val_size: int, image_size: list
) -> tuple:
    """
    Parameters:
        train_size: int - no of training sample
        test_size: int - no of training sample
        val_size: int - no of training sample
        image_size: tuple - image size ex. (50, 50, 1), here 1 is dim

    Description: generate (N X N) pixel representation for training, testing, and validation set.
                 Instead of taking only one non zero pixel and remaining all zero, I will inverse the condition
                 only 1 zero pixel and remaining all non-zero pixel replaced by 1.

    Returns:
        - tuple: return training i/p data, target i/p data, testing i/p data, testing target data, validation i/p data,
                 and validation target data
    """

    # create n dimension zero array
    train_images = np.ones(
        (train_size, image_size[0], image_size[1], image_size[2]), dtype=np.float32
    )
    train_coordinates = []
    test_images = np.ones(
        (test_size, image_size[0], image_size[1], image_size[2]), dtype=np.float32
    )
    test_coordinates = []
    val_images = np.ones(
        (val_size, image_size[0], image_size[1], image_size[2]), dtype=np.float32
    )
    val_coordinates = []

    for i in range(train_size):
        x = np.random.randint(0, image_size[0])
        y = np.random.randint(0, image_size[0])
        train_images[i][x][y] = 0                 # replacing 1 by 0
        train_coordinates.append([x, y])

    for i in range(test_size):
        a = np.random.randint(0, image_size[0])
        b = np.random.randint(0, image_size[0])
        test_images[i][a][b] = 0                 # replacing 1 by 0
        test_coordinates.append([a, b])

    for i in range(val_size):
        c = np.random.randint(0, image_size[0])
        d = np.random.randint(0, image_size[0])
        val_images[i][c][d] = 0                 # replacing 1 by 0
        val_coordinates.append([c, d])

    train_coordinates = np.array(train_coordinates).astype(np.float32)
    test_coordinates = np.array(test_coordinates).astype(np.float32)
    val_coordinates = np.array(val_coordinates).astype(np.float32)

    return (
        train_images,
        train_coordinates,
        test_images,
        test_coordinates,
        val_images,
        val_coordinates,
    )


if __name__ == "__main__":
    import pathlib
    import yaml

    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent.as_posix()
    params = yaml.safe_load(open(f"{home_dir}/params.yaml"))["make_dataset"]
    x_train, y_train, x_test, y_test, x_val, y_val = inverse_dataset(
        params["train_size"],
        params["test_size"],
        params["val_size"],
        params["image_size"],
    )
