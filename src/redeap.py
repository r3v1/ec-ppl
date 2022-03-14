"""
Reimplemented DEAP functions
"""
import random
import re
import sys
from inspect import isclass

import pyro
import torch
from pyro import distributions as tdist

from grammar import safeDiv, safePow  # , safeLog10


def genFull(pset, min_, max_, terminal_types, type_=None):
    """
    Generate an expression where each leaf has the same depth
    between *min* and *max*.
    
    Parameters
    ----------

    pset
        Primitive set from which primitives are selected.
    min_
        Minimum height of the produced trees.
    max_
        Maximum Height of the produced trees.
    type_
        The type that should return the tree when called, when
        `None` (default) the type of `pset` (pset.ret) is assumed.
        
    Returns
    -------
    A full tree with all leaves at the same depth.
    """
    
    # def condition(height, depth):
    #     """Expression generation stops when the depth is equal to height."""
    #     return depth == height
    
    # return generate(pset, min_, max_, condition, type_)
    return generate_safe(pset, min_, max_, terminal_types, type_)


def genGrow(pset, min_, max_, terminal_types, type_=None):
    """
    Generate an expression where each leaf might have a different depth
    between *min* and *max*.
    
    Parameters
    ----------

    pset
        Primitive set from which primitives are selected.
    min_
        Minimum height of the produced trees.
    max_
        Maximum Height of the produced trees.
    type_
        The type that should return the tree when called, when
                  :obj:`None` (default) the type of :pset: (pset.ret)
                  is assumed.
                  
    Returns
    -------
    A grown tree with leaves at possibly different depths.
    """
    
    # def condition(height, depth):
    #     """Expression generation stops when the depth is equal to height
    #     or when it is randomly determined that a node should be a terminal.
    #     """
    #     return depth == height or \
    #            (depth >= min_ and random.random() < pset.terminalRatio)
    
    # return generate(pset, min_, max_, condition, type_)
    return generate_safe(pset, min_, max_, terminal_types, type_)


def genHalfAndHalf(pset, min_, max_, terminal_types, type_=None):
    """
    Generate an expression with a PrimitiveSet *pset*.
    Half the time, the expression is generated with :func:`~deap.gp.genGrow`,
    the other half, the expression is generated with :func:`~deap.gp.genFull`.
    
    Parameters
    ----------

    pset
        Primitive set from which primitives are selected.
    min_
        Minimum height of the produced trees.
    max_
        Maximum Height of the produced trees.
    type_
        The type that should return the tree when called, when
                  :obj:`None` (default) the type of :pset: (pset.ret)
                  is assumed.
                  
    Returns
    -------
    Either, a full or a grown tree.
    """
    method = random.choice((genGrow, genFull))
    return method(pset, min_, max_, terminal_types, type_)


def generate_safe(pset, min_, max_, terminal_types, type_=None):
    """
    Safe version of deap.gp.generate function
    
    Notes
    -----
    Basically, when there are no terminals, it tries to put a primitive
    with 0 or more terminal only args instead of just a real terminal,
    and if that is not available, it defaults to putting a regular random
    primitive.
    Not sure if it's the correct way to go about things but it works for
    me and my use case.

    Note that using this makes height a recommendation instead of a hard
    limit, as there could be a case that the tree continues to expand beyond
    height (that's why I try to match primitives with only terminal args -
    to cut this possibly infinite branch generation, just like a terminal would).

    References
    ----------
    - https://github.com/DEAP/deap/issues/237#issuecomment-508087233
    """
    if type_ is None:
        type_ = pset.ret
    expr = []
    height = random.randint(min_, max_)
    stack = [(0, type_)]
    while len(stack) != 0:
        depth, type_ = stack.pop()
        
        if type_ in terminal_types:
            try:
                term = random.choice(pset.terminals[type_])
            except IndexError:
                _, _, traceback = sys.exc_info()
                raise IndexError("The gp.generate function tried to add "
                                 "a terminal of type '%s', but there is "
                                 "none available." % (type_,)).with_traceback(traceback)
            if isclass(term):
                term = term()
            expr.append(term)
        else:
            try:
                # Might not be respected if there is a type without terminal args
                if height <= depth or (depth >= min_ and random.random() < pset.terminalRatio):
                    primitives_with_only_terminal_args = [p for p in pset.primitives[type_] if
                                                          all([arg in terminal_types for arg in p.args])]
                    
                    if len(primitives_with_only_terminal_args) == 0:
                        prim = random.choice(pset.primitives[type_])
                    else:
                        prim = random.choice(primitives_with_only_terminal_args)
                else:
                    prim = random.choice(pset.primitives[type_])
            except IndexError:
                _, _, traceback = sys.exc_info()
                raise IndexError("The gp.generate function tried to add "
                                 "a primitive of type '%s', but there is "
                                 "none available." % (type_,)).with_traceback(traceback)
            expr.append(prim)
            for arg in reversed(prim.args):
                stack.append((depth + 1, arg))
    return expr


def compile_pyro(expr, init_args, replace: dict = None):
    code = str(expr)
    model = recurse(code, init_args, replace)
    
    return model


def recurse(expr, init_args, replace: dict = None):
    m = re.match(r'(\w+)\((.+)\)', expr)
    root = m.groups()[0]
    params = m.groups()[1:]
    
    args = []
    params_ = split_params(params[0])
    used_names = {}
    
    for p in params_:
        obj = final_recursion(p, init_args, 1, used_names, replace=replace)
        args.append(obj)
    
    root_dist = str2pyro(root, args, 0, used_names)
    
    return root_dist


def final_recursion(expr, init_args, depth: int, used_names: dict, replace: dict = None):
    m = re.match(r'(\w+)\((.+)\)', expr)
    root = m.groups()[0]
    params = m.groups()[1:]
    
    args = []
    params_ = split_params(params[0])
    
    if len(params_):
        for p in params_:
            obj = final_recursion(p, init_args, depth + 1, used_names, replace=replace)
            args.append(obj)
    else:
        # TODO: Cambiar esta forma de localizar el argumento de entrada, tal vez pasando el pset
        if init_args is None:
            # WARNING: No se ha contemplado toda la casuística
            if isinstance(params, (tuple, list)):
                arg = eval(params[0])
            elif isinstance(params, str):
                arg = eval(params)
            else:
                raise NotImplementedError()
            
            args.append(arg)
        else:
            if isinstance(init_args, (list, torch.Tensor)):
                if "x" == params[0]:
                    idx = 0
                elif "y" == params[0]:
                    idx = 1
                elif "z" == params[0]:
                    idx = 2
                else:
                    raise NotImplementedError()
            else:
                idx = params[0]
            args.append(init_args[idx])
    
    return str2pyro(root, args, depth, used_names, replace=replace)


def str2pyro(root: str, args, depth: int, used_names: dict, replace: dict = None):
    if depth not in used_names.keys():
        used_names[depth] = {}
    
    if root not in used_names[depth]:
        used_names[depth][root] = 0
    else:
        used_names[depth][root] += 1
    
    name = f"{root}_{depth}_{used_names[depth][root]}"
    
    if replace:
        for k in replace.keys():
            if k == name:
                return replace.pop(k)

    if root == "toTensor":
        try:
            obj = pyro.deterministic(name, torch.Tensor(*args))
        except IndexError:
            obj = pyro.deterministic(name, torch.Tensor(args))
    elif root == "Bernoulli":
        obj = tdist.Bernoulli(*args, validate_args=True)
    elif root == "Binomial":
        obj = tdist.Binomial(*args, validate_args=True)
    elif root == "Poisson":
        obj = tdist.Poisson(*args, validate_args=True)
    elif root == "Beta":
        obj = tdist.Beta(*args, validate_args=True)
    elif root == "Normal":
        obj = tdist.Normal(*args, validate_args=True)
    elif root == "Exponential":
        obj = tdist.Exponential(*args, validate_args=True)
    elif root == "Chi2":
        obj = tdist.Chi2(*args, validate_args=True)
    elif root == "toSample":
        obj = pyro.sample(name, *args)
    elif root == "toReal01Tensor":
        obj = pyro.deterministic(name, args[0].abs().clip(min=0, max=1))
    elif root == "toRealPositiveTensor":
        obj = pyro.deterministic(name, args[0].abs().clip(min=1e-7))
    elif root == "toNaturalTensor":
        obj = pyro.deterministic(name, args[0].abs().clip(min=1).round())
    elif root == "toNatural0Tensor":
        obj = pyro.deterministic(name, args[0].abs().round())
    elif root == "add":
        obj = pyro.deterministic(name, args[0] + args[1])
    elif root == "sub":
        obj = pyro.deterministic(name, args[0] - args[1])
    elif root == "mul":
        obj = pyro.deterministic(name, args[0] * args[1])
    elif root == "safePow":
        obj = pyro.deterministic(name, safePow(*args))
    elif root == "safeDiv":
        obj = pyro.deterministic(name, safeDiv(*args))
    # elif root == "safeLog10":
    #     obj = pyro.deterministic(name, safeLog10(*args))
    else:
        raise NotImplementedError(f"'{root}' not implemented")
    
    return obj


def split_params(text):
    """
    Splits text on one level depth brackets
    
    Parameters
    ----------
    text

    Returns
    -------
    list
    """
    i = 0
    opened = 0
    started = False
    start_idx = 0
    idxs = []
    while i < len(text):
        if text[i] == "(":
            opened += 1
            if not started:
                started = True
        elif text[i] == ")":
            opened -= 1
        
        i += 1
        
        if started and opened == 0:
            # Completed param specification
            idxs.append((start_idx, i))
            start_idx = i
            started = False
    
    return [text[slice(*idx)].strip(", ") for idx in idxs]


if __name__ == "__main__":
    from utils import load_temperature
    from pyro.infer import MCMC, NUTS
    
    obs = load_temperature()
    init_args = {"x": torch.rand(1), "y": torch.rand(1), "z": torch.rand(1)}
    
    
    def pyro_model(individual, obs):
        model = compile_pyro(individual, init_args)
        
        with pyro.plate("data"):
            pyro.sample("obs", model, obs=obs)
    
    
    def pyro_posterior(individual, num_samples: int = 100):
        kernel = NUTS(pyro_model, jit_compile=True, ignore_jit_warnings=True, max_tree_depth=10)
        posterior = MCMC(kernel, num_samples=num_samples, warmup_steps=int(num_samples * .25))
        
        posterior.run(individual, obs.squeeze())
        
        return posterior
    
    
    # individual = 'Chi2(toNaturalTensor(toSample(Normal(toTensor(z), toRealPositiveTensor(toTensor(y))))))'
    # individual = 'Normal(toTensor(z), toRealPositiveTensor(add(safePow(toRealPositiveTensor(toNaturalTensor(toTensor(y))), toTensor(y)), toSample(Normal(add(toTensor(z), toTensor(x)), toRealPositiveTensor(toTensor(z)))))))'
    # posterior = pyro_posterior(individual, num_samples=5)
    # samples = posterior.get_samples()
    # print(samples)
    individual = "Normal(toSample(Normal(toTensor([2]), toTensor([10]))), toRealPositiveTensor(toSample(Normal(toTensor([0]), toTensor([10])))))"
    
    samples = {'toSample_3_0': torch.Tensor([[-2.6488], [-2.6885], [-2.7873], [-2.7648], [-2.7632]])}
    model = compile_pyro(individual, None, replace=samples)
