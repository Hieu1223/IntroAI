<!-- GETTING STARTED -->
## Getting Started



### Prerequisites
Install required packets
  ```sh
  pip install -r requirements.txt
  ```

## Project Structure

* dataset/ – Contains the Ames Housing CSV file.

* linear_regression.py – Linear regression model training and evaluation.

* neural_network.py – Neural network model training and evaluation.
* ames_dataset.py - The dataset class that handles loading preprocessing data

* decision_tree.py – Decision tree model training and evaluation.

* random_forest.py – Random forest model training and evaluation.

* plot_gen.py - Contains function for plotting

* dataset_visualizer.py - Used to visualize and get statistics of raw data

* images/ – Generated plots for analysis 

* requirements.txt – Python dependencies for the project.


<!-- USAGE EXAMPLES -->
## Usage
After installing dependencies, you can run any model script:
  ```sh
python linear_regression.py
python neural_network.py
python decision_tree.py
python random_forest.py
```

Plots will be saved in the corresponding folder in images/