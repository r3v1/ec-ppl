import torch
from pyro import distributions as tdist


class Input: pass


class RealDistribution(tdist.distribution.Distribution):
    """
    Base class for continuous probability distributions
    sampling values within
    
    .. math::
        \mathbb{R} \in (-\infty, +\infty)
        
    e.g.: Normal distribution
    """
    pass


class NaturalDistribution(tdist.distribution.Distribution):
    """
    Base class for discrete probability distributions
    sampling values within
    
    .. math::
        \mathbb{N} \in (0, +\infty)
        
    """
    pass


class Natural0Distribution(NaturalDistribution):
    """
    `NaturalDistribution` implementing discrete probability
    distributions sampling values within
    
    .. math::
        \mathbb{N} \in [0, +\infty)
        
        
    e.g.: Binomial and Poisson distribution
    """
    pass


class Natural01Distribution(NaturalDistribution):
    """
    `NaturalDistribution` implementing discrete probability
    distributions sampling values within
    
    .. math::
        \mathbb{N} \in [0, 1]
        
        
    e.g.: Bernoulli distribution
    """
    pass


class RealPositiveDistribution(RealDistribution):
    """
    `RealDistribution` implementing continuous probability
    distributions sampling values within
    
    .. math::
        \mathbb{R} \in (0, +\infty)
    """
    pass


class RealPositive0Distribution(RealDistribution):
    """
    `RealDistribution` implementing continuous probability
    distributions sampling values within
    
    .. math::
        \mathbb{R} \in [0, +\infty)
        
    e.g.: Exponential and Chi2 distributions
    """
    pass


class Real01Distribution(RealDistribution):
    """
    `RealDistribution` implementing continuous probability
    distributions sampling values within
    
    .. math::
        \mathbb{R} \in [0, 1]
        
    e.g.: Beta distribution
    """
    pass


class NaturalTensor(torch.Tensor):
    """
    `torch.Tensor` whose values belong to the domain
    
    .. math::
    
        \mathbb{N} \in (0, +\infty)
    
    """
    pass


class Natural0Tensor(torch.Tensor):
    """
    `torch.Tensor` whose values belong to the domain
    
    .. math::
    
        \mathbb{N} \in [0, +\infty)
    
    """
    pass


class Natural01Tensor(torch.Tensor):
    """
    `torch.Tensor` whose values belong to the domain
    
    .. math::
    
        \mathbb{N} \in [0, 1]
    
    """
    pass


class RealPositiveTensor(torch.Tensor):
    """
    `torch.Tensor` whose values belong to the domain
    
    .. math::
    
        \mathbb{R} \in (0, +\infty)
    
    """
    pass


class RealPositive0Tensor(torch.Tensor):
    """
    `torch.Tensor` whose values belong to the domain
    
    .. math::
    
        \mathbb{R} \in [0, +\infty)
    
    """
    pass


class Real01Tensor(torch.Tensor):
    """
    `torch.Tensor` whose values belong to the domain
    
    .. math::
    
        \mathbb{R} \in [0, 1]
    
    """
    pass
