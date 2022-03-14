import configparser
from pathlib import Path
from typing import Tuple

import pyro
import torch
from pyro.infer import NUTS, MCMC

from logger import make_logger
from utils import stats_from


class Fitness:
    def __init__(self,
                 toolbox,
                 obs: torch.Tensor,
                 data_stats: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                 save_prefix: str = "",
                 working_dir: Path = None,
                 id_: str = None, ):
        self.toolbox = toolbox
        self.obs = obs.squeeze()
        self.data_stats = data_stats
        
        self.id_ = id_
        self._save_prefix = save_prefix
        
        # Load configuration
        self.config = configparser.ConfigParser()
        self.config.read(Path(__file__).parent / "settings.cfg")
        
        # Working directory
        self.working_dir = working_dir
        
        # Define logger
        self.fitness_logger = make_logger(self.__class__.__name__, working_dir=self.working_dir)
        
        self.f_iters = int(self.config["EVOLUTION"]["f_iters"])
        assert self.f_iters > 0, f"Number of iterations to compute the fitness must be > 0, got {self.f_iters}"
        
        self.n_samples = int(self.config["EVOLUTION"]["n_samples"])
    
    def fitness(self, individual):
        raise NotImplementedError()


class SimpleFitness(Fitness):
    def __init__(self, **kwargs):
        super(SimpleFitness, self).__init__(**kwargs)
    
    def fitness(self, individual):
        """
        Calcula el fitness de un individuo según la suma de los
        cuadrados de los estadísticos de las muestras que pueda generar
        comparado con los datos originales.

        Como el muestreo es aleatorio, se realiza varias veces este
        muestreo para coger una mejor aproximación
        """
        
        fitnesses = []
        best_fitness = torch.tensor(float("inf"))
        try:
            model = self.toolbox.compile(expr=individual)
            for _ in range(self.f_iters):
                # Sample values
                values = model.sample([self.n_samples]).ravel()
                
                # Sampled stats
                values_stats = stats_from(values)
                
                # Compute fitness
                f = torch.Tensor([(x - y) ** 2 for x, y in zip(self.data_stats, values_stats)]).sum()
                if not torch.isnan(f):
                    fitnesses.append(f)
        except IndexError as e:
            # IndexError: The gp.generate function tried to add a primitive
            # of type '<class 'src.gptypes.Natural01'>', but there is none available.
            self.fitness_logger.warning(str(e))
        except TypeError as e:
            if "expected TensorOptions" in str(e):
                self.fitness_logger.warning(str(e))
            else:
                raise e
        except ValueError as e:
            if "has invalid values" in str(e):
                self.fitness_logger.warning(str(e))
            else:
                raise e
        except Exception as e:
            self.fitness_logger.critical(str(e))
            exit(1)
        
        finally:
            if fitnesses:
                best_fitness = torch.mean(torch.Tensor(fitnesses))
        
        return best_fitness,


class MCMCFitness(Fitness):
    def __init__(self, mcmc_samples: int = 100, **kwargs):
        super(MCMCFitness, self).__init__(**kwargs)
        
        self.mcmc_samples = mcmc_samples
        assert mcmc_samples >= 10
    
    def _pyro_model(self, model: torch.distributions.Distribution):
        with pyro.plate("data"):
            pyro.sample("obs", model, obs=self.obs)
    
    def pyro_posterior(self, model: torch.distributions.Distribution):
        kernel = NUTS(self._pyro_model, jit_compile=True, ignore_jit_warnings=True, max_tree_depth=10)
        posterior = MCMC(kernel,
                         num_samples=self.mcmc_samples,
                         warmup_steps=int(self.mcmc_samples * .25),
                         # initial_params={
                         #     "tau_att": torch.Tensor([8]),
                         #     "tau_def": torch.Tensor([19]),
                         #     "home": torch.Tensor([0.29]),
                         #     "intercept": torch.Tensor([0.1]),
                         #     "att_t": torch.from_numpy(att_starting_points.values),
                         #     "def_t": torch.from_numpy(def_starting_points.values)
                         # },
                         disable_progbar=bool(self.config["INFERENCE"]["disable_progbar"])
                         )
        
        samples = {}
        try:
            posterior.run(model)
            samples = posterior.get_samples()
        except ValueError as e:
            if "within the support" in str(e):
                # TODO: Pyro - revisar el error
                # self.evolution_logger.warning(str(e))
                pass
            elif "Continuous inference" in str(e):
                # TODO: Pyro - revisar el error
                # self.evolution_logger.warning(str(e))
                pass
            elif "Error while packing" in str(e):
                # TODO: Pyro - revisar el error
                # self.evolution_logger.warning(str(e))
                pass
            elif "Model specification" in str(e):
                # TODO: Pyro - revisar el error
                # self.evolution_logger.warning(str(e))
                pass
            else:
                raise e
        except NotImplementedError as e:
            if "Inhomogeneous total" in str(e):
                # TODO: Pyro - revisar el error
                # self.evolution_logger.warning(str(e))
                pass
            else:
                raise e
        except RuntimeError as e:
            if "The size of tensor" in str(e):
                # TODO: Error en la función safePow
                pass
            else:
                raise e
        
        return samples
    
    def fitness(self, individual):
        """
        Calcula el fitness de un individuo según la suma de los
        cuadrados de los estadísticos de las muestras que pueda generar
        comparado con los datos originales.

        Primero se compila el modelo y se transforma en un modelo de Pyro,
        para luego inferir los parámetros y poder muestrear. En caso de no
        ser un modelo jerárquico, no se realiza inferencia y los parámetros
        de entrada son los predefinidos (en la constructora)
        """
        
        fitness = torch.tensor(float("inf"))
        try:
            # Transform the tree expression in a Pyro distribution function
            model = self.toolbox.compile(expr=individual)
            samples = self.pyro_posterior(model)
            
            # Recompile model with posterior
            model = self.toolbox.compile(expr=individual, replace=samples)
            
            # Sample values
            values = model.sample()
            
            # Sampled stats
            values_stats = stats_from(values)
            
            # Compute fitness
            fitness = torch.Tensor([(x - y) ** 2 for x, y in zip(self.data_stats, values_stats)]).sum()
        except TypeError as e:
            # TypeError: len() of a 0-d tensor
            self.fitness_logger.warning(str(e))
        except ValueError as e:
            if "has invalid values" in str(e):
                self.fitness_logger.warning(str(e))
            else:
                raise e
        except IndexError as e:
            # IndexError: The gp.generate function tried to add a primitive
            # of type '<class 'src.gptypes.Natural01'>', but there is none available.
            self.fitness_logger.warning(str(e))
        except RuntimeError as e:
            self.fitness_logger.warning(str(e))
        
        return fitness,
