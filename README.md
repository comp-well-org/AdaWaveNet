# AdaWaveNet

AdaWaveNet is a comprehensive framework for time series forecasting, imputation, and super-resolution tasks. 

Please refer to the paper for more details.

https://openreview.net/forum?id=m4bE9Y9FlX

```
@article{yu2025adawavenet,
  title={AdaWaveNet: Adaptive Wavelet Network for Time Series Analysis},
  author={Yu, Han and Guo, Peikun and Sano, Akane},
  journal={Transactions on Machine Learning Research},
  year={2025}
}
```

## Features

- **Long-term and Short-term Forecasting**: Supports models like Autoformer, Transformer, TimesNet, and more.
- **Imputation**: Handles missing data in time series.
- **Super Resolution**: Enhances the resolution of time series data.

## Requirements

The project requires the following Python packages, which can be installed using the `requirements.txt` file:


## Usage

The main entry point for running experiments is the `run.py` script. It supports various command-line arguments to configure the experiments. Here is an example of how to run a long-term forecasting task:

```
python -u run.py \
--task_name long_term_forecast \
--is_training 1 \
--root_path ./dataset/weather/ \
--data_path weather.csv \
--model_id weather_96_96 \
--model AdaWaveNet \
--data custom \
--features M \
--seq_len 96 \
--label_len 48 \
--pred_len 96 \
--e_layers 3 \
--d_layers 1 \
--factor 3 \
--enc_in 21 \
--dec_in 21 \
--c_out 21 \
--des 'Exp' \
--d_model 512 \
--d_ff 512 \
--itr 1 \
--lifting_levels 3 \
--lifting_kernel_size 7 \
--n_cluster 4 \
--learning_rate 0.0005 \
--batch_size 16
```

## Configuration

The `run.py` script accepts various arguments to configure the experiment:

- `--task_name`: The name of the task (e.g., long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection).
- `--is_training`: Whether to train the model (1 for training, 0 for testing).
- `--model`: The model to use (e.g., Autoformer, Transformer, TimesNet).
- `--seq_len`, `--label_len`, `--pred_len`: Sequence lengths for input, label, and prediction.
- `--e_layers`, `--d_layers`: Number of encoder and decoder layers.
- `--learning_rate`: Learning rate for the optimizer.
- `--batch_size`: Batch size for training.

For a full list of arguments, refer to the `run.py` script.


## Acknowledgments

This project is based on the Time-Series-Library Repository <https://github.com/thuml/Time-Series-Library> and other state-of-the-art time series models.

## License
AdaWaveNet is released under the GPL 3.0. See [LICENSE](https://github.com/comp-well-org/AdaWaveNet/blob/main/LICENSE.txt) for details.

