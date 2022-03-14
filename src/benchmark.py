import argparse
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tikzplotlib
from matplotlib import pyplot as plt
from torch import Tensor

from drawings import compare
from evolution import Evolution
from utils import check_cuda
from utils import load_temperature, load_precipitation, load_normal, load_beta


class Benchmark:
    def __init__(self,
                 title: str,
                 prefix: str,
                 generations: int,
                 X_train: Tensor,
                 X_test: Tensor = None,
                 id_: str = None,
                 savefig: bool = True,
                 loss: str = "simple",
                 optimizer: str = None):
        self.working_dir = Path("~/.evolution").expanduser()
        self.generations = generations

        self.title = title
        self.prefix = prefix
        self.X_train = X_train
        self.X_test = X_test

        # Evolution results
        self.evolution_results = {
            "raw": None,
            "normalize": None,
            "standardize": None
        }
        
        self.id_ = id_ or f"{datetime.datetime.now():%Y%m%d_%H%M%S}"
        self.savefig = savefig
        self.loss = loss
        self.optimizer = optimizer
        
        # Just for info
        check_cuda()
    
    def _launch(self, scaling: str = None):
        print("#" * 80)
        t = f'Launching benchmark: {self.title}'
        print(f"# {t:^74} #")
        print("#" * 80, end="\n\n")
        
        if scaling is None or scaling == "raw":
            save_prefix = f"{self.prefix}_raw_"
        elif scaling == "normalize":
            save_prefix = f"{self.prefix}_norm_"
        elif scaling == "standardize":
            save_prefix = f"{self.prefix}_std_"
        else:
            raise ValueError(f"`{scaling}` scaling type not found.")
        
        ##########################
        #    Evolution process   #
        ##########################
        output_path = self.working_dir / self.title
        
        e = Evolution(self.X_train,
                      working_dir=output_path,
                      save_prefix=save_prefix,
                      generations=self.generations,
                      scaling=scaling,
                      id_=self.id_,
                      savefig=self.savefig,
                      loss=self.loss,
                      optimizer=self.optimizer)
        e.evolve()
        e.save_evolution_log()
        e.save_best()

        if self.prefix == "driver":
            # Compare with observed only with drivers problem (too few observed cases)
            model, scaler = self.evolution_results[scaling]
            compare(self.X_test,
                    model,
                    output_path / self.id_ / "test",
                    self.id_,
                    save_prefix,
                    scaler=scaler,
                    n_samples=5000,
                    savetikz=True)
        ##########################

        print("#" * 80)
        t = f'Finished: {self.title}'
        print(f"# {t:^74} #")
        print("#" * 80)

        # Return best individual, id and scaler for testing purposes
        self.evolution_results[scaling] = e.best_pyro_model, e.scaler

        # Important deleting the instance
        e.__del__()

    def fit(self):
        for scaling in ["raw", "normalize", "standardize"]:
            self._launch(scaling=scaling)

    def test(self, n_samples: int = 5000):
        """
        Just checks if the best evolved individual is capable
        of generalizing any drawn sample from the known model
        """

        for scaling in ["raw", "normalize", "standardize"]:
            if scaling is None or scaling == "raw":
                save_prefix = f"{self.prefix}_raw_"
            elif scaling == "normalize":
                save_prefix = f"{self.prefix}_norm_"
            elif scaling == "standardize":
                save_prefix = f"{self.prefix}_std_"
            else:
                raise ValueError(f"`{scaling}` scaling type not found.")

            model, scaler = self.evolution_results[scaling]

            if self.X_test is not None:
                compare(self.X_test,
                        model,
                        self.working_dir / self.title / self.id_ / "test",
                        self.id_,
                        save_prefix,
                        scaler=scaler,
                        n_samples=n_samples,
                        savetikz=True)
    
    def compile_results(self):
        outdir = self.working_dir / self.title
        
        id_ = '_'.join(self.id_.split('_')[:-1])
        
        for scaling in ["raw", "norm", "std"]:
            plot_outfile = f"{id_}_{self.prefix}_{scaling}_logbook.png"

            d = outdir.glob(f"{id_}_*/{id_}_*_{scaling}_logbook.csv")
            
            # Join all logbooks
            df = None
            for rep, filename in enumerate(d):
                df_ = pd.read_csv(filename, index_col="gen")
                df_.columns = [f"{x}_{rep}" for x in df_.columns]
                
                if df is None:
                    df = df_.copy()
                else:
                    df = pd.concat([df, df_], axis=1)
            
            if df is not None:
                # Save logbook compilation
                df.to_csv(outdir / f"{id_}_{self.prefix}_{scaling}_logbook.csv")
                
                # Plot
                df_ = df.filter(regex="(avg|max|min)")
                # df_ = df_.where(df_ < df_.quantile(0.9))
                
                # avg = df_.filter(regex="avg").mean(axis=1)
                # max_ = df_.filter(regex="max").mean(axis=1)
                min_ = df_.filter(regex="min").mean(axis=1)
                
                with plt.style.context(['science', 'ieee', 'grid']):
                    fig, ax = plt.subplots(1, 1, figsize=(19.2 / 4, 10.8 / 4))
                    
                    # ax1.plot(avg, color="b", label="$avg$")
                    # ax1.fill_between(df_.index, min_, max_, alpha=0.2, label="$min-max$")
                    # ax1.set_ylabel(f"Fitness")
                    # ax1.grid(True)
                    # # ax1.legend()
                    
                    ax.plot(min_, color="black")
                    df_.filter(regex="min").plot(legend=False, ax=ax, alpha=0.65, lw=0.45)
                    
                    # minimum = df_.filter(regex="min").min(axis=1)
                    # minimum_idx = np.argmin(minimum)
                    # ax2.scatter(minimum_idx, minimum.min(), marker="x", color="black", zorder=30, s=10)
                    # ax2.annotate(f"${minimum.min().round(4)}$",
                    #              (minimum_idx, minimum.min()),
                    #              bbox=dict(boxstyle="round", fc="w"),
                    #              va="center",
                    #              ha="left",
                    #              xytext=(100, 100),
                    #              textcoords="offset pixels",
                    #              arrowprops=dict(arrowstyle="->"),
                    #              fontsize=4,
                    #              )
                    
                    plt.ylabel(f"Fitness")
                    plt.xlabel("Generation")
                    plt.grid(True, alpha=.05)
                    
                    plt.tight_layout()
                    plt.savefig(self.working_dir / self.title / plot_outfile, dpi=256)
                    
                    phi = 1.618033988
                    scale = 0.75
                    tikzplotlib.save(outdir / plot_outfile.replace(".png", ".tex"),
                                     axis_width=f'{scale:.3}\linewidth',
                                     axis_height=f'{scale / phi:.3}\linewidth')
                    
                    # plt.show()
                    plt.close()


class TemperatureBench(Benchmark):
    def __init__(self, **kwargs):
        title = "Problem 2.1: Average Minimum Temperature in Scotland"
        X = load_temperature()
        
        # Holdout
        idx = np.random.permutation(len(X))
        X_train = X[idx[:int(len(idx) * .65)]]
        X_test = X[-idx[int(len(idx) * .65):]]
        assert len(X_train) + len(X_test) == len(X)
        
        super().__init__(title=title,
                         prefix="temperature",
                         X_train=X_train,
                         X_test=X_test,
                         **kwargs)


class PrecipitationBench(Benchmark):
    def __init__(self, **kwargs):
        title = "Precipitation modeling"
        X = load_precipitation()
        
        # Holdout
        idx = np.random.permutation(len(X))
        X_train = X[idx[:int(len(idx) * .65)]]
        X_test = X[-idx[int(len(idx) * .65):]]
        assert len(X_train) + len(X_test) == len(X)
        
        super().__init__(title=title,
                         prefix="precipitation",
                         X_train=X_train,
                         X_test=X_test,
                         **kwargs)


class Normal(Benchmark):
    def __init__(self, **kwargs):
        title = "Normal distribution modeling"
        
        # Original data to train with
        X = load_normal()
        
        # Holdout
        idx = np.random.permutation(len(X))
        X_train = X[idx[:int(len(idx) * .65)]]
        X_test = X[-idx[int(len(idx) * .65):]]
        assert len(X_train) + len(X_test) == len(X)
        
        super().__init__(title=title,
                         prefix="normal",
                         X_train=X_train,
                         X_test=X_test,
                         **kwargs)


class Beta(Benchmark):
    def __init__(self, **kwargs):
        title = "Beta distribution modeling"
        
        # Original data to train with
        X = load_beta()
        
        # Holdout
        idx = np.random.permutation(len(X))
        X_train = X[idx[:int(len(idx) * .65)]]
        X_test = X[-idx[int(len(idx) * .65):]]
        assert len(X_train) + len(X_test) == len(X)
        
        super().__init__(title=title,
                         prefix="beta",
                         X_train=X_train,
                         X_test=X_test,
                         **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark")
    parser.add_argument("--bench", help="'temperature': Launches the `2.1 Average Minimum Temperature in Scotland` "
                                        "problem benchmark.\n'precipitation': Models the precipitation in Punta Galea",
                        choices=["temperature", "precipitation", "normal", "beta"], required=True)
    parser.add_argument("--generations", "-g", help="Number of generations to evolve (default: 100)", default=100)
    parser.add_argument("--id",
                        help="Folder name to store results (default: current datetime)",
                        default=f"{datetime.datetime.now():%Y%m%d_%H%M%S}")
    parser.add_argument("-r", "--repeat", help="Repeats n times the experiment (default: 1)", default=1, type=int)
    parser.add_argument("--no-figures", help="Doesn't save figures", action="store_true")
    parser.add_argument("-l", "--loss", help="Select the loss function: \n'simple' performs parameter inference only on"
                                             " the best evolved individual. 'inference', inference is applied to every "
                                             "single individual in the evolution (slower). (Default: simple)",
                        choices=["simple", "inference"], default="simple")
    parser.add_argument("-o", "--optimizer",
                        help="Select the input optimizer: \n'svi' performs SVI posterior distribution"
                             " ont the best evolved individual. 'botorch', numerical optimization (not implemented yet)"
                             "(Default: None)",
                        choices=["svi", "botorch"], default=None)
    parser.add_argument("--continue-from", help="Continues experiment number n (default: 0)", default=0, type=int)

    args = parser.parse_args()

    n_generations = int(args.generations)

    benchs = {
        "temperature": TemperatureBench,
        "precipitation": PrecipitationBench,
        "normal": Normal,
        "beta": Beta,
    }
    
    bench = None
    for rep in range(args.continue_from, max(args.repeat, args.continue_from + 1)):
        # Check the las index and adds 1
        
        id_ = f"{args.id}_{rep}"
        kwargs = dict(
            generations=n_generations,
            id_=id_,
            savefig=not args.no_figures,
            loss=args.loss,
            optimizer=args.optimizer,
        )

        bench = benchs[args.bench](**kwargs)
        bench.fit()
        bench.test()

    # Compile logbook CSV results
    id_ = f"{args.id}_0"
    kwargs = dict(
        generations=n_generations,
        # use_model=args.use_model,
        id_=id_,
        savefig=not args.no_figures,
        loss=args.loss,
        optimizer=args.optimizer,
    )

    bench = benchs[args.bench](**kwargs)
    bench.compile_results()
