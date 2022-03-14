import configparser
import datetime
import operator
from pathlib import Path
from typing import Union

import joblib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import tikzplotlib
from deap import algorithms, base, creator, gp, tools
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from bfs import plot_tree, simplify
from fitness import SimpleFitness, MCMCFitness
from gptypes import *
from grammar import grammar
from logger import make_logger
from optimizer import SVIOptimizer
from redeap import genFull, genHalfAndHalf, compile_pyro
from utils import stats_from

torch.autograd.set_detect_anomaly(True)


class Evolution:
    def __init__(self,
                 data,
                 save_prefix: str = "",
                 n_individuals: int = None,
                 generations: int = None,
                 working_dir: Path = None,
                 scaling: str = None,
                 id_: str = None,
                 savefig: bool = True,
                 loss: str = "simple",
                 optimizer: str = None):
        """
        Initialize Evolution object parameters
        
        Parameters
        ----------
        data: list like or numpy array
            Observed data to build the model
        save_prefix: str
            Prepend a string to all saved files.
            Default: ""
        n_individuals: int, None
            Number of individuals in the population.
            Defaults to `settings.cfg` file.
        generations: int, None
            Number of generations to perform de evolution
            Defaults to `settings.cfg` file.
        working_dir: Path, None
            Working directory (a.k.a the folder in which
            generated files will be stored).
            Defaults to `settings.cfg` file.
        scaling: str, None
            `standardize` applies StandardScaler and
            `normalize` applies MinMaxScaler.
            Defaults to no scaling.
        id_: str, None
            File identifier also used as subdirectory.
            Defaults to current datetime.
        loss: str
            If 'simple', parameter inference is performed only
            on the best evolved individual. If 'inference', inference
            is applied to every single individual in the evolution
        """
        ################################
        #         System config
        ################################
        # Just to identify files
        self.id_ = id_ or f"{datetime.datetime.now():%Y%m%d_%H%M%S}"
        self._save_prefix = save_prefix
        self.savefig = savefig
        
        # Load configuration
        self.config = configparser.ConfigParser()
        self.config.read(Path(__file__).parent / "settings.cfg")
        
        # Working directory
        self.working_dir = working_dir or Path(self.config["EVOLUTION"]["working_dir"]).expanduser()
        self.working_dir /= self.id_
        if not self.working_dir.is_dir():
            self.working_dir.mkdir(parents=True, exist_ok=True)
        
        # Define logger
        self.evolution_logger = make_logger(self.__class__.__name__, working_dir=self.working_dir)
        
        # Input arguments
        self.n_samples = int(self.config["EVOLUTION"]["n_samples"])
        n = 1  # len(data)
        assert n > 0, f"Samples to draw must be > 0, got {n}"
        self._args = [
            torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([1])).sample(),
            torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([1])).sample(),
            torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([1])).sample()
        ]
        
        # Retrieve original statistics from data
        self._original_data = data
        self.original_data_stats = stats_from(self._original_data)
        
        # Use scaling
        self.scaler = None
        self._scaled_data = None
        self.scaled_data_stats = None
        if scaling and scaling != "raw":
            if scaling == "standardize":
                # StandardScaler
                self.scaler = StandardScaler()
            elif scaling == "normalize":
                # MinMaxScaler
                self.scaler = MinMaxScaler((1e-7, 1 - 1e-7))
            else:
                raise ValueError(f"`{scaling}` scaling type not found.")

            self.evolution_logger.info(f"{scaling.capitalize()[:-1]}ing original data")
            self._scaled_data = torch.from_numpy(self.scaler.fit_transform(data.reshape(-1, 1)))
            self._scaled_data = torch.reshape(self._scaled_data, self._original_data.shape)
            self.scaled_data_stats = stats_from(self._scaled_data)
        
        ################################
        #          DEAP stuff
        ################################
        
        # Initialize grammar
        type_ = tdist.distribution.Distribution
        pset = grammar(type_=type_)
        
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, pset=pset)
        
        # Evolution process
        self.toolbox = base.Toolbox()
        
        min_ = int(self.config["EVOLUTION"]["min"])
        assert min_ > 0, f"Minimum depth must be > 0, got {min_}"
        max_ = int(self.config["EVOLUTION"]["max"])
        assert max_ > min_, f"Maximum depth must be max > {min_} > 0, got max={max_} and min={min_}"
        
        self.toolbox.register("expr", self._genHalfAndHalf, pset=pset, min_=min_, max_=max_,
                              terminal_types=[Input] * n)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        
        self.n_individuals = n_individuals or int(self.config["EVOLUTION"]["n_individuals"])
        assert self.n_individuals > 0, f"Individuals to evolve must be > 0, got {self.n_individuals}"
        self.verbose = bool(self.config["EVOLUTION"]["verbose"])
        self.generations = generations or int(self.config["EVOLUTION"]["generations"])
        assert self.generations > 0, f"Generations to evolve must be > 0, got {self.generations}"
        self.toolbox.register("population", self._generate_population, self.toolbox.individual)
        # self.toolbox.register("compile", gp.compile, pset=pset)
        self.toolbox.register("compile", compile_pyro, init_args=self._args)
        
        tournsize = int(self.config["EVOLUTION"]["tournsize"])
        assert tournsize > 0, f"Tournament size must be > 0, got {tournsize}"
        self.toolbox.register("select", tools.selTournament, tournsize=tournsize)
        
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", self._genFull, min_=0, max_=3, terminal_types=[Input] * n)
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr_mut, pset=pset)
        
        heigh_limit = int(self.config["EVOLUTION"]["heigh_limit"])
        assert heigh_limit > 0, f"heigh_limit must be > 0, got {heigh_limit}"
        self.toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=heigh_limit))
        self.toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=heigh_limit))
        
        # Compute stats from individuals
        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        self.mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        self.mstats.register("avg", lambda x: f"{torch.Tensor(x).mean().item():.4e}")
        # self.mstats.register("std", np.std)
        self.mstats.register("min", lambda x: f"{torch.Tensor(x).min().item():.4f}")
        self.mstats.register("max", lambda x: f"{torch.Tensor(x).max().item():.4e}")
        
        # Evolution containers
        self.hof = None
        self.population = None
        self.logbook = None
        self.best_pyro_model = None
        
        # Init fitness object
        fitness_kwargs = dict(toolbox=self.toolbox,
                              obs=self._scaled_data if self._scaled_data is not None else self._original_data,
                              data_stats=self.scaled_data_stats or self.original_data_stats,
                              save_prefix=save_prefix,
                              id_=self.id_,
                              working_dir=self.working_dir)
        if loss == "inference":
            self.fitness = MCMCFitness(mcmc_samples=int(self.config["INFERENCE"]["MCMC_samples"]), **fitness_kwargs)
        else:
            self.fitness = SimpleFitness(**fitness_kwargs)
        
        self.toolbox.register("evaluate", self.fitness.fitness)
        
        # Refine best results
        self.optimum_inputs = None
        optimizer_kwargs = dict(toolbox=self.toolbox,
                                obs=self._scaled_data if self._scaled_data is not None else self._original_data,
                                input_args=self._args,
                                save_prefix=save_prefix,
                                id_=self.id_,
                                working_dir=self.working_dir)
        assert optimizer in [None, "", "svi", "botorch"], f"Optimizer '{optimizer}' not in ['svi', 'botorch']"
        if optimizer == "svi":
            self.optimizer = SVIOptimizer(**optimizer_kwargs)
        elif optimizer == "botorch":
            raise NotImplementedError(f"BoTorch not implemented")
        else:
            self.optimizer = None
    
    def __del__(self):
        try:
            del creator.Individual
            del creator.FitnessMin
        except AttributeError:
            pass
    
    def _generate_population(self, individual_maker, n):
        """
        Overrides `tools.initRepeat` from DEAP
        """
        pop = []
        illegals = 0
        while len(pop) < n:
            try:
                i = individual_maker()
                pop.append(i)
            except IndexError as e:
                illegals += 1
        
        if illegals:
            self.evolution_logger.warning(f"{illegals} illegal individuals built")
        
        return pop
    
    def _genHalfAndHalf(self, pset, min_, max_, terminal_types, type_=None):
        """
        Overrides `gp.genHalfAndHalf` from DEAP
        
        Notes
        -----
        Used to generate individuals
        """
        try:
            return genHalfAndHalf(pset=pset, min_=min_, max_=max_, terminal_types=terminal_types, type_=type_)
        except IndexError as e:
            self.evolution_logger.warning(str(e))
            return self._genHalfAndHalf(pset=pset, min_=min_, max_=max_, terminal_types=terminal_types, type_=type_)
    
    def _genFull(self, pset, min_, max_, terminal_types, type_=None):
        """
        Overrides `gp.genFull` from DEAP
        
        Notes
        -----
        Used to generate mutated individuals
        """
        try:
            return genFull(pset=pset, min_=min_, max_=max_, terminal_types=terminal_types, type_=type_)
        except IndexError as e:
            self.evolution_logger.warning(str(e))
            return self._genFull(pset=pset, min_=min_, max_=max_, terminal_types=terminal_types, type_=type_)
    
    def evolve(self):
        """
        Launch evolution
        """
        self.evolution_logger.info(f"Evolving population with id `{self.id_}`")
        
        # Initialize population
        self.population = self.toolbox.population(n=self.n_individuals)
        self.hof = tools.HallOfFame(10)
        
        # Actually evolve
        self.population, self.logbook = algorithms.eaSimple(self.population,
                                                            self.toolbox,
                                                            0.5,
                                                            0.1,
                                                            self.generations,
                                                            stats=self.mstats,
                                                            halloffame=self.hof,
                                                            verbose=self.verbose)
        
        # Learn posterior for the best evolved individual
        self.get_best_as_pyro()
    
    def save_evolution_log(self):
        """
        Saves the logbook
        """
        if self.logbook:
    
            gen = self.logbook.select("gen")
            fit_mins = list(map(float, self.logbook.chapters["fitness"].select("min")))
            size_avgs = list(map(float, self.logbook.chapters["size"].select("avg")))
    
            # Save CSV
            df = pd.DataFrame(self.logbook.chapters["fitness"]).set_index("gen")
            df.to_csv(self.working_dir / f"{self.id_}_{self._save_prefix}logbook.csv")
    
            if self.savefig:
                outfile = self.working_dir / f"{self.id_}_{self._save_prefix}logbook.png"
                self.evolution_logger.info(f"Saving evolution logbook plot to {outfile}")
                with plt.style.context(['science', 'ieee', 'grid']):
                    fig, ax1 = plt.subplots(figsize=(19.8 / 4, 10.8 / 4))
                    line1 = ax1.plot(gen, fit_mins, "-", color="black", label="Minimum\ Fitness")
                    ax1.set_xlabel("Generation")
                    ax1.set_ylabel("Fitness", color="black")
                    for tl in ax1.get_yticklabels():
                        tl.set_color("black")
                    
                    ax2 = ax1.twinx()
                    line2 = ax2.plot(gen, size_avgs, "-", color="red", label="Average\ Size")
                    ax2.set_ylabel("Size", color="red")
                    for tl in ax2.get_yticklabels():
                        tl.set_color("red")
                    
                    sc = ax1.scatter(np.argmin(fit_mins), min(fit_mins), c='blue',
                                     label=f"Best:\ {min(fit_mins):.4f}", s=10, marker="x", zorder=10)

                    lns = line1 + line2 + [sc]
                    labs = [l.get_label() for l in lns]
                    plt.legend(lns, labs,
                               # loc="center right",
                               prop={'size': 5},
                               )

                    plt.grid()
                    plt.tight_layout()

                    tikzfile = outfile.parent / outfile.name.replace(".png", ".tex")
                    phi = 1.618033988
                    scale = 0.75
                    tikzplotlib.save(tikzfile.as_posix(),
                                     # extra_axis_parameters=["dashed"],
                                     axis_width=f'{scale:.3}\linewidth',
                                     axis_height=f'{scale / phi:.3}\linewidth')

                    plt.savefig(outfile, dpi=256)
                plt.close()
        else:
            self.evolution_logger.warning(f"Empty logbook. Run `evolve` first")
    
    def get_best_as_pyro(self):
        """
        Returns the best evolved individual as a pyro distribution,
        with posterior learnt
        """
        if self.hof:
            individual = self.hof[0]

            if self.optimizer:
                try:
                    self.optimum_inputs = self.optimizer.optimize(individual)
                except ValueError:
                    # This exception is usually triggered when the observed values are in
                    # a domain outside the scope of the model. In principle, it is not our
                    # problem, but that of evolution, which has not learned too well, I guess?
        
                    # self.evolution_logger.warning(str(e))
                    self.optimum_inputs = None
                    model = self.toolbox.compile(expr=individual, replace=self.optimum_inputs)
        
                    self.evolution_logger.warning("Could not be optimized because the observed values are "
                                                  f"outside the domain of the learned distribution: {model.support}")

            # Recompile model with posterior
            self.best_pyro_model = self.toolbox.compile(expr=individual, replace=self.optimum_inputs)

        else:
            self.evolution_logger.critical(f"Empty hall of fame. Run `evolve` first")
    
    def save_best(self):
        def plot_tree_aux(nodes: Union[list, set],
                          edges: list,
                          labels: dict,
                          title: str,
                          outfile: Path,
                          samples: dict,
                          savetikz: bool = True):
            g = nx.Graph()
            g.add_nodes_from(nodes)
            g.add_edges_from(edges)

            for k, v in labels.items():
                if v == "x":
                    v = f"{samples['x'].item():.4f}" if samples else f"{self._args[0].item():.4f}"
                elif v == "y":
                    v = f"{samples['y'].item():.4f}" if samples else f"{self._args[1].item():.4f}"
                elif v == "z":
                    v = f"{samples['z'].item():.4f}" if samples else f"{self._args[2].item():.4f}"

                labels[k] = v

            plot_tree(g, title=title, labels=labels, savefig=outfile, savetikz=savetikz)

        if self.hof:
            if self.savefig:
                outfile = self.working_dir / f"{self.id_}_{self._save_prefix}best_full.png"
                outfile_simple = self.working_dir / f"{self.id_}_{self._save_prefix}best.png"
                self.evolution_logger.info(f"Plotting best individual to {outfile}")

                # Get best inputs parameters
                # Save entire individual to text file
                with open(self.working_dir / f"{self.id_}_{self._save_prefix}best.txt", "w") as fp:
                    str_model = str(self.hof[0])
                    if self.optimum_inputs:
                        str_model = str_model.replace("(x)", f"([{self.optimum_inputs['x'].item():.4}])")
                        str_model = str_model.replace("(y)", f"([{self.optimum_inputs['y'].item():.4}])")
                        str_model = str_model.replace("(z)", f"([{self.optimum_inputs['z'].item():.4}])")
                    else:
                        str_model = str_model.replace("(x)", f"([{self._args[0].item():.4}])")
                        str_model = str_model.replace("(y)", f"([{self._args[1].item():.4}])")
                        str_model = str_model.replace("(z)", f"([{self._args[2].item():.4}])")
                    fp.write(str_model)

                # Save Pyro model
                pyro_output = self.working_dir / f"{self.id_}_{self._save_prefix}best_pyro.joblib"
                joblib.dump(self.best_pyro_model, pyro_output)

                nodes, edges, labels = gp.graph(self.hof[0])
                plot_tree_aux(nodes,
                              edges,
                              labels,
                              title=rf"$Best\ individual:\ {self.hof[0].fitness.values[0]:.4f}$",
                              outfile=outfile,
                              samples=self.optimum_inputs,
                              savetikz=True)

                # Simplify individual plot
                nodes, edges, labels = simplify(nodes, edges, labels)
                plot_tree_aux(nodes,
                              edges,
                              labels,
                              title=rf"$Best\ individual:\ {self.hof[0].fitness.values[0]:.4f}$",
                              outfile=outfile_simple,
                              samples=self.optimum_inputs,
                              savetikz=True)
        else:
            self.evolution_logger.critical(f"Empty hall of fame. Run `evolve` first")


if __name__ == '__main__':
    from utils import load_temperature
    
    original_data = load_temperature()
    
    # Normalizado [0, 1]
    scaling = "normalize"
    
    # Estandarizado
    # scaling = "standardize"
    
    e = Evolution(original_data, generations=5, scaling=scaling, optimizer="svi")
    e.evolve()
    e.save_evolution_log()
    # e.compare()
    e.save_best()
