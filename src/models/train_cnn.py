import numpy as np
from src.models.logger import logger
from tensorflow.keras import layers, models  # type: ignore


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
) -> float:
    """
    Parameters:
        x_train: np.ndarray - training input data
        y_train: np.ndarray - training target data
        
        x_test: np.ndarray - testing input data
        y_test: np.ndarray - testing target data
        
        x_val: np.ndarray - validation input data
        y_val: np.ndarray - validation target data
        
        epochs: int - no of epochs

    Returns:
        - float: MAE of test data
    """
    # Define the model
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dense(2))

    # Compile the model
    model.compile(
        optimizer="adam", loss="mean_squared_error", metrics=["mean_absolute_error"]
    )

    # fit the model on training data
    history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_val, y_val))

    # log the training performnace
    logger(history, file_name="train")

    # Evaluate the model
    test_loss, test_mae = model.evaluate(x_test, y_test, verbose=2)

    # Make predictions
    # predictions = model.predict(x_test)
    # return the test loss
    return round(test_mae, 4)


if __name__ == "__main__":
    import yaml
    import pathlib
    from src.data.make_dataset import make_dataset

    # load params.yaml to access parameters
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent.as_posix()
    params = yaml.safe_load(open(f"{home_dir}/params.yaml"))

    # create a dataset
    x_train, y_train, x_test, y_test, x_val, y_val = make_dataset(
        params["make_dataset"]["train_size"],
        params["make_dataset"]["test_size"],
        params["make_dataset"]["val_size"],
        params["make_dataset"]["image_size"],
    )

    # train the model
    mae = train_model(
        x_train, y_train, x_test, y_test, x_val, y_val, params["train_cnn"]["epochs"]
    )

    print(mae)
