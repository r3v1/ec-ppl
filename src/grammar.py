# For adding DEAP primitives
import numbers
import operator
from typing import Union

import torch
from deap import gp

from gptypes import *


def safeDiv(left: Union[float, torch.Tensor], right: Union[float, torch.Tensor]) -> torch.Tensor:
    """
    Protected divison to avoid `ZeroDivisionError`

    Parameters
    ----------
    left: Union[float, torch.Tensor]
        Numerator
    right: Union[float, torch.Tensor]
        Denominator

    Returns
    -------
    torch.Tensor
    """
    result = left / right
    result[right == 0] = 0
    return result


def safePow(a: Union[float, list, torch.Tensor], b: Union[float, list, torch.Tensor]) -> torch.Tensor:
    """
    Protected pow function to avoid exponentiation of
    negative numbers to fractional numbers, rounding
    exponents to nearest integer

    Parameters
    ----------
    a: float, list, torch.Tensor
        Base
    b: float, list, torch.Tensor
        Exponent

    Returns
    -------
    torch.Tensor
    """

    def _check_tensor(x) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            if isinstance(x, (list, tuple)):
                x = torch.tensor(x)
            elif isinstance(x, (float, int)):
                x = torch.tensor([x])
            else:
                raise NotImplementedError()
        else:
            # Already a torch.Tensor
            if x.ndim == 0:
                x = torch.tensor([x.clone().detach()])

        return x

    # Cast to torch.Tensor
    a = _check_tensor(a)
    b = _check_tensor(b)

    if len(a.size()) == len(b.size()):
        # Find negative numbers indexes
        idxs = torch.where(a < 0)
        # Round negative exponents to nearest integer
        b[idxs] = torch.round(b[idxs])

    elif 1 == len(a.size()) < len(b.size()):
        if a[0] < 0:
            # Find negative exponents indexes
            idxs = torch.where(b < 0)
            # Round negative exponents to nearest integer
            b[idxs] = torch.round(b[idxs])

    elif 1 == len(b.size()) < len(a.size()):
        if b[0] < 0:
            # Find negative exponents indexes
            idxs = torch.where(a < 0)
            # Round negative exponents to nearest integer
            a[idxs] = torch.round(b[idxs])

    elif 1 < len(a.size()) < len(b.size()):
        # Similar error to torch.pow when
        # torch.pow(torch.Tensor([2, 0.5554]),torch.Tensor([0.9072, 2, 3]))
        raise RuntimeError(f"The size of tensor a ({len(a)}) must match the size of tensor b ({len(b)}) "
                           f"at non-singleton dimension 0")
    else:
        raise NotImplementedError(f"Error at 'pow': len(a) = {len(a)}, len(b) = {len(b)}")

    result = torch.pow(a, b)
    if result.isnan().any():
        raise NotImplementedError()

    if result.isinf().any():
        result = result.clamp(min=-1e18, max=1e18)

    return result


# def safeLog10(a: Union[float, list, torch.Tensor]) -> torch.Tensor:
#     a[a > 0] = torch.log10(a[a > 0])
#
#     return a


def grammar(type_: type = tdist.distribution.Distribution) -> gp.PrimitiveSetTyped:
    """
    Defines the primitive typed set of the grammar

    Parameters
    ----------
    type_: tdist.distribution.Distribution
        Return type

    Returns
    -------
    gp.PrimitiveSetTyped
    """
    pset = gp.PrimitiveSetTyped("tree", [Input, Input, Input], type_)
    
    #######################################################
    #             Define discrete distributions
    #######################################################
    def Bernoulli(p) -> Natural01Distribution:
        pass
    
    def Binomial(n, p) -> Natural0Distribution:
        pass
    
    def Poisson(l) -> Natural0Distribution:
        pass
    
    #######################################################
    #           Define continuous distributions
    #######################################################
    def Beta(alpha, beta) -> Real01Distribution:
        pass
    
    def Normal(mu, sigma) -> RealDistribution:
        pass
    
    def Exponential(l) -> RealPositive0Distribution:
        pass
    
    def Chi2(k) -> RealPositive0Distribution:
        pass
    
    #######################################################
    #                   Define operators
    #######################################################
    def toSample(dist: tdist.distribution.Distribution) -> torch.Tensor:
        pass
    
    def toTensor(x: Union[numbers.Number, torch.Tensor, list, set]) -> torch.Tensor:
        pass
    
    def toReal01Tensor(x: torch.Tensor) -> Real01Tensor:
        """
        Converts values to

        .. math::

            \mathbb{R} \in [0, 1]

        domain

        Parameters
        ----------
        x: torch.Tensor

        Returns
        -------
        Real01Tensor
        """
        pass
    
    def toRealPositiveTensor(x: torch.Tensor) -> RealPositiveTensor:
        """
        Converts values to

        .. math::

            \mathbb{R} \in (0, +\infty)

        domain

        Parameters
        ----------
        x: torch.Tensor

        Returns
        -------
        RealPositiveTensor
        """
        pass
    
    # def toRealPositive0Tensor(x: torch.Tensor) -> RealPositive0Tensor:
    #     """
    #     Converts values to
    #
    #     .. math::
    #
    #         \mathbb{R} \in [0, +\infty)
    #
    #     domain
    #
    #     Parameters
    #     ----------
    #     x: torch.Tensor
    #
    #     Returns
    #     -------
    #     RealPositive0Tensor
    #     """
    #     pass
    
    def toNaturalTensor(x: torch.Tensor) -> NaturalTensor:
        """
        Converts values to

        .. math::

            \mathbb{N} \in (0, +\infty)

        domain

        Parameters
        ----------
        x: torch.Tensor

        Returns
        -------
        NaturalTensor
        """
        pass
    
    def toNatural0Tensor(x: torch.Tensor) -> Natural0Tensor:
        """
        Converts values to

        .. math::

            \mathbb{N} \in [0, +\infty)

        domain

        Parameters
        ----------
        x: torch.Tensor

        Returns
        -------
        Natural0Tensor
        """
        pass
    
    #######################################################
    #              Adding to the primitive set
    #######################################################
    pset.addPrimitive(toSample, [tdist.distribution.Distribution], torch.Tensor)
    
    # Distributions
    pset.addPrimitive(Bernoulli, [Real01Tensor], Natural01Distribution)
    pset.addPrimitive(Binomial, [Natural0Tensor, Real01Tensor], Natural0Distribution)
    pset.addPrimitive(Poisson, [RealPositiveTensor], Natural0Distribution)
    
    pset.addPrimitive(Beta, [RealPositiveTensor, RealPositiveTensor], Real01Distribution)
    pset.addPrimitive(Normal, [torch.Tensor, RealPositiveTensor], RealDistribution)
    pset.addPrimitive(Exponential, [RealPositiveTensor], RealPositive0Distribution)
    pset.addPrimitive(Chi2, [NaturalTensor], RealPositive0Distribution)
    
    # Unary operators
    pset.addPrimitive(toReal01Tensor, [torch.Tensor], Real01Tensor)
    pset.addPrimitive(toRealPositiveTensor, [torch.Tensor], RealPositiveTensor)
    # pset.addPrimitive(toRealPositive0Tensor, [torch.Tensor], RealPositive0Tensor)
    pset.addPrimitive(toNaturalTensor, [torch.Tensor], NaturalTensor)
    pset.addPrimitive(toNatural0Tensor, [torch.Tensor], Natural0Tensor)
    
    # Binary operators
    pset.addPrimitive(operator.add, [torch.Tensor, torch.Tensor], torch.Tensor)
    pset.addPrimitive(operator.sub, [torch.Tensor, torch.Tensor], torch.Tensor)
    pset.addPrimitive(operator.mul, [torch.Tensor, torch.Tensor], torch.Tensor)
    pset.addPrimitive(safePow, [torch.Tensor, torch.Tensor], torch.Tensor)
    pset.addPrimitive(safeDiv, [torch.Tensor, torch.Tensor], torch.Tensor)
    # pset.addPrimitive(safeLog10, [torch.Tensor], torch.Tensor)
    
    pset.addPrimitive(toTensor, [Input], torch.Tensor)
    
    pset.renameArguments(ARG0="x")
    pset.renameArguments(ARG1="y")
    pset.renameArguments(ARG2="z")
    
    # Add terminals
    # Con esto salvamos los casos (algunos) en lo que la generación del árbol llegue a
    # su profundidad máxima sin haber acabado en un terminal, lo que DEAP obligará
    # a utilizar un terminal que devuelva un tipo que no existe para acabar el árbol. Lo
    # que le decimos aquí es que agrupe varias funciones de una sóla vez, así desde un 'float'
    # podemos llegar a distintos tipos en un sólo nodo (terminal)
    # for arg in pset.arguments:
    #     pset.addTerminal(lambda: toReal01Tensor(toTensor(arg)), Real01Tensor, f"toReal01Tensor(toTensor({arg}))")
    #     pset.addTerminal(lambda: toRealPositiveTensor(toTensor(arg)), RealPositiveTensor,
    #                      f"toRealPositiveTensor(toTensor({arg}))")
    #     pset.addTerminal(lambda: toRealPositive0Tensor(toTensor(arg)), RealPositive0Tensor,
    #                      f"toRealPositive0Tensor(toTensor({arg}))")
    #     pset.addTerminal(lambda: toNaturalTensor(toTensor(arg)), NaturalTensor, f"toNaturalTensor(toTensor({arg}))")
    #     pset.addTerminal(lambda: toNatural0Tensor(toTensor(arg)), Natural0Tensor, f"toNatural0Tensor(toTensor({arg}))")
    #
    #     pset.addTerminal(lambda: toTensor(arg), torch.Tensor, f"toTensor({arg})")
    #     pset.addTerminal(arg, Input, arg)
    
    return pset
