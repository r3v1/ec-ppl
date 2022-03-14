import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from deap import gp


def random_string() -> str:
    """
    Generate random string

    References
    ----------
    - https://stackoverflow.com/a/58915982
    """
    return np.random.randint(low=97, high=122, size=10, dtype="int32").view(f"U10")[0]


def stats_from(array: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calcula estadísticos de un array dado

    Note
    ----
    Casos extremos como `[1., 1., 1.]`, darán como resultado
    una varianza 0 y por lo tanto, la asimetría y la curtósis
    no se podrán computar

    References
    ----------
    - https://discuss.pytorch.org/t/statistics-for-whole-dataset/74511/2
    """
    assert type(array) == torch.Tensor, f"Not a torch.Tensor object"
    
    mean = torch.mean(array)
    diffs = array - mean
    var = torch.mean(torch.pow(diffs, 2.0))
    std = torch.pow(var, 0.5)
    
    # Avoid dividing by 0
    if 0 <= std.item() <= 10e-15:
        std = 10e-15
    elif -10e-15 <= std.item() <= 0:
        std = -10e-15
    
    zscores = diffs / std
    skews = torch.mean(torch.pow(zscores, 3.0))
    kurtoses = torch.mean(torch.pow(zscores, 4.0)) - 3.0
    
    return mean, var, skews, kurtoses


def load_temperature() -> torch.Tensor:
    """
    Loads the `Average Minimum Temperature in Scotland` problem

    Returns
    -------
    torch.Tensor

    References
    ----------
    - https://eprints.gla.ac.uk/226085/1/226085.pdf
    - https://www.metoffice.gov.uk/pub/data/weather/uk/climate/datasets/Tmin/date/Scotland.txt
    """
    return torch.Tensor([
        1.1, 1.1, 2.5, 0.7, 2.7, 2.8, 1.3, 1.0, 2.2, 0.3, 3.6, 1.7,
        2.1, 3.5, 1.4, 4.5, 2.3, 1.2, 3.4, 1.7, 1.3, 1.1, 3.7, 1.4,
        2.9, -0.3, -1.1, 1.0, 1.5, 3.2, 1.8, -1.3, 3.2, 3.1, 0.8,
        -1.6, 3.6, 0.7, 2.1, -0.5, 3.4, -0.1, 1.1, 1.8, 2.7, 2.0,
        0.6, 3.6, 1.8, 1.6, 1.4, 1.7, 1.3, 1.5, 3.8, 3.0, 1.9, 1.7,
        1.0, 2.1, 0.4, 3.4, 2.7, 1.6, 3.0, 2.4, 0.6, 3.4, -0.3, 4.0,
        1.5, 3.5, 2.6, 2.7, 2.2, 2.8, 1.7, 1.1, 1.4, 2.1, 2.3, -0.2,
        0.5, 1.4, 1.7, -0.8, 1.9, 1.6, 0.9, 0.8, 1.7, 1.7, 1.5, 0.9,
        3.1, 1.2, 1.9, 2.5, 2.2, 3.0, 3.3, -1.1, 2.5, 2.6, 1.7, 2.0,
        2.0, 2.0, 1.5, 0.4, 5.1, 3.0, 0.2, 4.6, 1.3, 2.9, 1.8, 3.0,
        3.8, 3.3, 3.3, 1.6, 3.3, 3.4, 2.0, 2.8, -0.0, 4.8, 1.7, 1.1,
        3.7, 3.5, 0.2, 1.5, 3.5, 1.1, 3.9
    ])


def load_reaction_times(day: int = 6) -> torch.Tensor:
    """
    Loads `Drivers Reaction Times` problem data

    Parameters
    ----------
    day: int
        Day to filter. Options: 0 - 9

    Returns
    -------
    torch.Tensor

    Notes
    -----
    Eighteen lorry drivers were chosen and were restricted to 3 hours
    of sleep during the trial. Their reaction time to a visual stimulus
    was measured on each day of the experiment for 10 days. A simple model
    that could model the variation in reaction times of the  lorry drivers
    across the days of the sleep-deprivation is a linear regression model
    of the following form: the  reaction time for a driver $i,i ∈ [0, 17]$
    on day $t,t ∈ [0, 9]$ would follow a normal distribution with a mean
    value floating on a straight line of the form $ai + t · bi$. A greater
    reaction time on  every next day of the sleep-deprivation trial is
    anticipated. Thus, a straight line with positive slope would be an
    appropriate model for modelling the reaction time of the drivers across
    the days of sleep-deprivation.

    References
    ----------
    - https://eprints.gla.ac.uk/226085/1/226085.pdf
    - https://github.com/amir1m/BayesianStatistics/blob/master/evaluation_sleepstudy.csv
    """
    assert day in list(range(0, 10))
    
    reactions = pd.read_csv(Path(__file__).parents[1] / "data" / "evaluation_sleepstudy.csv",
                            usecols=['Reaction', 'Days', 'Subject'])
    driver_idx, drivers = pd.factorize(reactions["Days"], sort=True)
    
    # DIMENSIONS
    # coords = {"driver": drivers, "driver_idx_day": reactions.Subject}
    
    # Utilizar los datos de un sólo día?
    data = torch.Tensor(reactions[reactions.Days == day]["Reaction"].values)
    
    return data


def load_volatility() -> torch.Tensor:
    """
    Loads ` Stochastic Volatility` problem data

    Returns
    -------
    torch.Tensor

    Notes
    -----
    The aim of the stochastic volatility model is to model
    the day returns of assets. Assets prices are variable
    and they have a time-varying volatility. There are periods,
    when returns are highly variable, while in other
    periods they are more stable. This variability is modeled
    by using a latent volatility variable.

    References
    ----------
    - https://eprints.gla.ac.uk/226085/1/226085.pdf

    See Also
    --------
    - Dataset downloaded from https://raw.githubusercontent.com/pymc-devs/pymc-examples/main/examples/data/SP500.csv
    """
    
    returns = pd.read_csv(Path(__file__).parents[1] / "data" / "SP500.csv",
                          parse_dates=True,
                          index_col="Date")
    returns["change"] = np.log(returns["Close"]).diff()
    returns = returns.dropna()
    
    # Get only change
    data = torch.Tensor(returns["change"].values)
    
    return data


def expr2individual(expr: str) -> gp.PrimitiveTree:
    from grammar import grammar
    
    pset = grammar()
    individual = gp.PrimitiveTree.from_string(expr, pset)
    
    return individual


def load_coal() -> torch.Tensor:
    """
    Loads ` Coal Mining Disasters` problem data

    Returns
    -------
    torch.Tensor

    Notes
    -----
    The aim of the coal mining disasters model is to predict the time point in the past when the number of
    the recorded coal mining-related disasters in the UK started to decline. It is assumed that the decline in
    the number of the coal mining-related disasters was linked to changes in the safety regulations. The data
    set that we used provides the time series of recorded coal mining disasters in the UK from 1851 to 1962
    (Jarrett, 1979)

    References
    ----------
    - https://eprints.gla.ac.uk/226085/1/226085.pdf
    """
    
    disaster_data = torch.Tensor(
        [4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6, 3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5, 2, 2, 3, 4, 2,
         1, 3,
         # np.nan,
         2, 1, 1, 1, 1, 3, 0, 0, 1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 2,
         1, 0, 0, 0, 1, 1, 0, 2, 3, 3, 1,
         # np.nan,
         2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0,
         0, 1, 0, 1])
    years = np.arange(1851, 1962)
    years_missing = [1890, 1935]

    return disaster_data


def load_normal() -> torch.Tensor:
    filename = Path(__file__).parents[1] / "data" / "normal.pt"
    x = torch.load(filename.as_posix())
    
    return x


def load_beta() -> torch.Tensor:
    filename = Path(__file__).parents[1] / "data" / "beta.pt"
    x = torch.load(filename.as_posix())
    
    return x


def check_cuda():
    print('__Python VERSION:', sys.version)
    print('__pyTorch VERSION:', torch.__version__)
    print('__CUDA VERSION', )
    # from subprocess import call
    # # call(["nvcc", "--version"]) does not work
    # ! nvcc --version
    print('__CUDNN VERSION:', torch.backends.cudnn.version())
    print('__Number CUDA Devices:', torch.cuda.device_count())
    try:
        cd = torch.cuda.current_device()
        print('__Devices')
        # call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
        print('Active CUDA Device: GPU', cd)
        print('Available devices ', torch.cuda.device_count())
        print('Current cuda device ', cd)
    except RuntimeError:
        pass


def load_precipitation() -> torch.Tensor:
    station_id = "C042"
    df = pd.read_csv(Path(__file__).parents[1] / "data" / f"{station_id}.csv", index_col="DATE", parse_dates=["DATE"])
    df = df.filter(regex="^Precip").copy()
    df.rename(columns={"Precip.._a_140cm": "PRECIP"}, inplace=True)
    df.dropna(inplace=True)
    
    # Lluvia acumulada en 24 horas
    df = df.resample("24H").sum()
    
    x = torch.from_numpy(df["PRECIP"].values)
    
    return x


if __name__ == "__main__":
    check_cuda()
