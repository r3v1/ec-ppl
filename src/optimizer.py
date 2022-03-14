import configparser
from pathlib import Path
from typing import Union

import pyro
from tqdm import tqdm

from drawings import plot_elbo, plot_latent_values
from gptypes import *
from logger import make_logger

torch.autograd.set_detect_anomaly(True)


class Optimizer:
    def __init__(self,
                 toolbox,
                 obs: torch.Tensor,
                 input_args: Union[list, torch.Tensor],
                 save_prefix: str = "",
                 working_dir: Path = None,
                 id_: str = None, ):
        self.toolbox = toolbox
        self.obs = obs.squeeze()
        self.input_args = input_args
        assert len(input_args) > 0
        
        self.id_ = id_
        self._save_prefix = save_prefix
        
        # Load configuration
        self.config = configparser.ConfigParser()
        self.config.read(Path(__file__).parent / "settings.cfg")
        
        # Working directory
        self.working_dir = working_dir
        
        # Define logger
        self.optimizer_logger = make_logger(self.__class__.__name__, working_dir=self.working_dir)
    
    def optimize(self, indivual) -> dict:
        raise NotImplementedError()


class SVIOptimizer(Optimizer):
    def __init__(self, **kwargs):
        super(SVIOptimizer, self).__init__(**kwargs)
        
        # Inference
        self.svi_samples = None
        self.svi_steps = int(self.config["OPTIMIZER"]["svi_steps"])
        assert self.svi_steps >= 0, f"SVI steps must be >= 0"
    
    def optimize(self, individual) -> dict:
        def gp_conditioned(params, evidence):
            weights = pyro.sample("weight", tdist.Normal(params, torch.ones(3)).to_event(1))
            m = self.toolbox.compile(expr=individual, init_args=weights)
            with pyro.plate("data", len(evidence)):
                pyro.sample("measurement", m, obs=evidence)
        
        def guide(params, evidence):
            latent = pyro.param("latent", params)
            pyro.sample("weight", tdist.Normal(latent, torch.ones(3)).to_event(1))

        self.optimizer_logger.info(f"Learning posterior for the best evolved individual")

        pyro.clear_param_store()
        svi = pyro.infer.SVI(model=gp_conditioned,
                             guide=guide,
                             optim=pyro.optim.Adam({"lr": 0.01}),
                             loss=pyro.infer.Trace_ELBO())

        losses = torch.zeros(self.svi_steps, dtype=torch.float64)
        latent_values = torch.zeros((self.svi_steps, len(self.input_args)))

        trange = tqdm(range(self.svi_steps), desc=f"SVI posterior distribution")
        for t in trange:
            losses[t] = svi.step(torch.Tensor(self.input_args), self.obs)
            latent_values[t] = pyro.param("latent")

        # Show inference logs
        self.optimizer_logger.info(f"Saving SVI plots")
        plot_elbo(losses, self.working_dir / f"{self.id_}_{self._save_prefix}ELBO_loss.png")

        plot_latent_values(latent_values,
                           self.input_args,
                           self.working_dir / f"{self.id_}_{self._save_prefix}latent_variables.png")
        
        # print()
        # print("Last latent values")
        # print('x = ', pyro.param("latent")[0])
        # print('y = ', pyro.param("latent")[1])
        # print('z = ', pyro.param("latent")[2])
        # print()
        # print("Last 10 mean latent values")
        # print('x = ', latent_values[-10:, 0].mean())
        # print('y = ', latent_values[-10:, 1].mean())
        # print('z = ', latent_values[-10:, 2].mean())
        # print()
        
        return {k: latent_values[-10:, v].mean() for k, v in zip(("x", "y", "z"), range(3))}
