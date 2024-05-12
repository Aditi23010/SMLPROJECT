# Project Name

This project contains code for training and testing models for image classification.

## Folder Structure

- **Model**: Contains saved model parameters.
- **Data**: Contains images and annotations for train, validation, and testing.
- **Test**: Contains scripts to load pre-trained models and test them against the test dataset.
- **Code**: Contains training files for ResNet and AlexNet models.
    - ResNet.ipynb: Training file for ResNet.
    - AlexNet.ipynb: Training file for AlexNet.
- **Requirement.txt**: Lists all dependencies required to run the code.
- **Report.pdf**: Report of the project.

## Running the Project

To run the project:

1. Open a terminal in the root folder.
2. Make sure Python version 3.8.8 is installed.
3. Install all dependencies from `Requirement.txt`:

    ```bash
    pip install -r Requirement.txt
    ```

4. Change the directory to `Test`:

    ```bash
    cd Test
    ```

5. Run the model scripts to test them against the test data:

    ```bash
    python "modelname.py"
    ```

    For example:

    ```bash
    python ResNet.py
    python AlexNet.py
    ```

## Additional Notes

- Ensure that the necessary data files are present in the `Data` folder before running the test scripts.
- Modify the scripts as needed to specify the correct paths to data and model files.
