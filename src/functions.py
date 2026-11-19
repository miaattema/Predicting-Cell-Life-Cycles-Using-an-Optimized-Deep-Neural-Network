import sys
import numpy as np

def progress_bar(c_iter, n_iters, n_updates=1000, length_bar=20):
    update_every = max(1, round(n_iters / n_updates))
    if c_iter % update_every == 0:
        progress = c_iter / n_iters
        length_fill = round(length_bar * progress)
        length_void = length_bar - length_fill
        sys.stdout.write(f"\r{progress:6.0%} [" + length_fill * "=" + length_void * " " + "]")

def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
    iqr = np.quantile(y_true, 0.75, axis=0) - np.quantile(y_true, 0.25, axis=0)
    nrmse = np.divide(rmse, iqr, out=np.full_like(rmse, np.nan), where=iqr != 0)
    return rmse, nrmse
