## Requirements

Requires Python 3.8+. Install dependencies (virtual environment recommended) with:

```
pip install -U -r requirements.txt
```

If `pip` command is linked to a Python 2 environment, try:

```
python3 -m pip install -U -r requirements.txt
```

## Execution

```
usage: benchmark.py [-h] --bench {temperature,drivers,precipitation,normal,beta} [--day {0,1,2,3,4,5,6,7,8,9}] [--generations GENERATIONS] [--id ID] [-r REPEAT] [--no-figures]
                    [-l {simple,inference}] [-o {svi,botorch}] [--continue-from CONTINUE_FROM]

Benchmark

optional arguments:
  -h, --help            show this help message and exit
  --bench {temperature,drivers,precipitation,normal,beta}
                        'temperature': Launches the `2.1 Average Minimum Temperature in Scotland` problem benchmark. 
                        'precipitation': Models the precipitation in Punta Galea
  --generations GENERATIONS, -g GENERATIONS
                        Number of generations to evolve (default: 100)
  --id ID               Folder name to store results (default: current datetime)
  -r REPEAT, --repeat REPEAT
                        Repeats n times the experiment (default: 1)
  --no-figures          Doesn't save figures
  -l {simple,inference}, --loss {simple,inference}
                        Select the loss function: 
                        'simple' performs parameter inference only on the best evolved individual. 
                        'inference', inference is applied to every single individual in the evolution (slower). (Default: simple)
  -o {svi,botorch}, --optimizer {svi,botorch}
                        Select the input optimizer: 'svi' performs SVI posterior distribution on the best evolved individual. 
                        'botorch', numerical optimization (not implemented yet) (Default: None)
  --continue-from CONTINUE_FROM
                        Continues experiment number n (default: 0)
```

## Executed experiments

Here are shown the shell commands executed for the experiments shown in the memory.

### Problem 2.1

**Before proceeding with these experiments, it is important to exclude Normal and Beta distributions respectively from
the primitive set before launching them.**

```shell
# Evolves during 250 generations the 'Normal' experiment with 'simple' loss function, repeating 5 times the entire process
python src/benchmark.py --bench normal --repeat 5 --generations 250 --id svi --loss simple --optimizer svi

# Evolves during 250 generations the 'Beta' experiment with 'simple' loss function, repeating 5 times the entire process
python src/benchmark.py --bench beta --repeat 5 --generations 250 --id svi --loss simple --optimizer svi
``` 

### Problem 2.2

```shell
# Evolves during 250 generations the 'Temperature' experiment with 'simple' loss function, repeating 5 times the entire process
python src/benchmark.py --bench temperature --repeat 5 --generations 250 --id svi --loss simple --optimizer svi
``` 

### Problem 2.3

```shell
# Evolves during 250 generations the 'Precipitation' experiment with 'simple' loss function, repeating 5 times the entire process
python src / benchmark . py -- bench precipitation -- repeat 5 --generations 250 -- id svi -- loss simple -- optimizer svi

``` 

## TODO

### Problemas

- [x] Solucionar el [problema](https://github.com/DEAP/deap/issues/237) por que no se podían añadir terminales.
- [x] No se puede restringir la raíz del programa a que saque una distribución concreta.
- [x] Inspeccionar el motivo de por qué los fitnesses devuelven NaNs.
- [x] Utilizar abreviaciones o acrónimos en el plot en vez de utilizar el nombre completo de la función.
- [x] Escalar de nuevo los datos a su dominio natural una vez evolucionado el mejor individuo.
- [x] Saber por qué la anchura del histograma es distinta para los datos observados y los datos muestreados.
- [x] Limpiar warnings `ValueError: The parameter scale has invalid values`. Parece que el motivo esta en el operador
  pow por defecto:
  ```
  b = tensor([ 0.0626, -0.5554,  1.5460, -0.4619])
  e = tensor([0.5808, 0.9072, 0.9675, 0.6521])
  b ** e
  # tensor([0.2001,    nan, 1.5243,    nan])
  
  # Utilizando la función 'torch.pow` también:
  torch.pow(torch.Tensor([-0.5554]),torch.Tensor([0.9072]))
  # tensor([nan])
  
  # Utilizando todos los dígitos y usando Torch:
  b[1].item() ** e[1].item()
  # (-0.5618083795592397+0.16863684749123004j)
  
  # Y utilizando puro Python:
  -0.5554**0.9072
  # -0.586551905740603
  ```
- [x] Eliminar del análisis el efecto de la varianza: fijar un mismo conjunto de test dentro de esas 100 repeticiones
  que hago cuando conocemos el modelo.
- [x] Definir holdout.
- [x] Probar todo este sistema con el MNIST.
- [x] Tensores de más de una dimensión funcionan en la evolución, el problema esta en la visualización (
  método `compare`)
  del histograma, habría que generar gráficas distintas para cada dimensión.

### Mejoras

- [x] Definir, por cada problema, un modelo conocido que se ajuste bien y ver cómo se comporta la evolución y si es
  capaz de llegar al mismo resultado o algo parecido.
- [x] Generar un benchmark.
- [x] Generar una batería de más problemas.
- [ ] Plantear una función multiobjetivo, penalizando los árboles profundos.

# Problemas con la Inferencia

Lo que he hecho para tratar todos estos errores ha sido omitirlos, y devolver un diccionario vacío si no se consigue
realizar la inferencia. Es decir, entiendo que el modelo puede seguir *funcionando* y generando valores, pero no
estará *optimizado*, ¿puede ser?. Este tratamiento de excepciones esta incluído dentro de la función `pyro_posterior()`.
Si esta suposición es incorrecta, y un modelo del que no se pueden inferir sus posteriores no debiera *existir*,
entonces el tratamiento de errores habría que sacarlo fuera, a la función de evaluación `_simple_fitness()`.

## 1. *The value argument must be within the support*

Una distribución/individuo cuyo soporte no admita los valores observados, dará error. Suponer el siguiente modelo
evolucionado:

```
Chi2(toNaturalTensor(toSample(Normal(toTensor(z), toRealPositiveTensor(toTensor(y))))))
```

y los siguientes valores observados:

```python
obs = [1.1000, 1.1000, 2.5000, 0.7000, 2.7000, 2.8000, 1.3000, 1.0000,
       2.2000, 0.3000, 3.6000, 1.7000, 2.1000, 3.5000, 1.4000, 4.5000,
       2.3000, 1.2000, 3.4000, 1.7000, 1.3000, 1.1000, 3.7000, 1.4000,
       2.9000, -0.3000, -1.1000, 0.000]
```

La variable `obs` tiene elementos negativos y ceros, valores no incluídos en el soporte de la distribución `Chi2`. Por
ejemplo, vamos a definir el modelo y ejecutar la inferencia

```python
from pyro import distributions as tdist
import pyro
import torch
from pyro.infer import MCMC, NUTS

init_args = {"x": torch.rand(137), "y": torch.rand(137), "z": torch.rand(137)}


def pyro_model(obs):
    normal = tdist.Normal(loc=init_args['z'], scale=init_args['y'].abs().clip(min=10e-7), validate_args=True)
    k = pyro.sample(f"toSample_0", normal)
    model = tdist.Chi2(df=k.abs().clip(min=1).round(), validate_args=True)
    
    with pyro.plate("data"):
        pyro.sample("obs", model, obs=obs)


def pyro_posterior(num_samples: int = 100):
    kernel = NUTS(pyro_model, jit_compile=True, ignore_jit_warnings=True, max_tree_depth=10)
    posterior = MCMC(kernel, num_samples=num_samples, warmup_steps=int(num_samples * .25))
    
    posterior.run(obs)
    
    return posterior
```

Al ejecutar la inferencia con,

```python
posterior = pyro_posterior(num_samples=100)
```

recibimos el siguiente error:

```
ValueError: Error while computing log_prob_sum at site 'obs':
The value argument must be within the support
  Trace Shapes:      
   Param Sites:      
  Sample Sites:      
toSample_0 dist 137 |
          value 137 |
       obs dist 137 |
          value 137 |
```

Pero si sustuimos dentro de la función `pyro_model` la última línea:

```python
with pyro.plate("data"):
    # pyro.sample("obs", model, obs=obs)    
    pyro.sample("obs", model, obs=obs.abs() + .01)
```

haciendo que los valores observados entren dentro del soporte de la distribución `Chi2`, la inferencia se ejecuta
correctamente, devolviendo los parámetros que andamos buscando:

```
Sample: 100%|██████████| 125/125 [01:06,  1.88it/s, step size=1.78e-02, acc. prob=0.549]

{'toSample_0': tensor([[ 0.8701, -1.3739,  1.8715,  ..., -0.7652,  1.4052,  1.6738],
        [ 0.6573, -1.6748,  1.5623,  ..., -0.4228,  1.3047,  1.7979],
        [ 0.6535, -1.6016,  1.6123,  ..., -0.3994,  1.3343,  1.8232],
        ...,
        [ 0.4374, -0.0531,  2.4454,  ...,  0.8106,  1.3718,  0.8478],
        [ 0.3755,  0.1236,  2.2230,  ...,  0.6915,  1.3393,  0.8100],
        [ 0.3996,  0.2877,  1.9811,  ...,  0.9667,  1.4918,  0.8328]])}
```

### ¿Soluciones?

- Se podría orientar la evolución a que generara individuos con una distribución raíz que sea capaz de muestrear valores
  de la misma naturaleza que los observados. Por ejemplo, para el caso anterior, restringiríamos la búsqueda a que todos
  los individuos tuvieran en la raíz una `Normal`.
    - **Contras**: Me dijiste al principio del proyecto que no se podía hacer así porque sería influenciar en la
      evolución

- Encapsularlo en un `try-except` y si recibimos el error, asignarle un fitness extremo, indicando que dicho individuo
  no es válido.
    - **Pros**: Se distinguen los individuos que son capaces de muestrear valores de la misma naturaleza que los
      observados

[comment]: <> (    - **Contras**: Puede haber muchos individuos no válidos y ralentizar la evolución)

## 2. *Value is not broadcastable with batch_shape+event_shape*

El tamaño de los vectores de entrada tiene que coincidir con el número de observaciones. Si no, la inferencia devuelve
el siguiente error:

```
ValueError: Error while computing log_prob_sum at site 'obs':
Value is not broadcastable with batch_shape+event_shape: torch.Size([89]) vs torch.Size([5000]).
            Trace Shapes:            
             Param Sites:            
            Sample Sites:            
        toTensor_2_0 dist      | 5000
                    value      | 5000
toNatural0Tensor_1_0 dist      | 5000
                    value      | 5000
        toTensor_2_1 dist      | 5000
                    value      | 5000
  toReal01Tensor_1_0 dist      | 5000
                    value      | 5000
                 obs dist 5000 |     
                    value   89 |
```

### Solución

Basta con cambiar en la constructura de `Evolution`:

```python
# Input arguments
# n = int(self.config["EVOLUTION"]["n_args"])
n = len(data)
```

## 3. *Continuous inference cannot handle discrete sample site*

No tengo muy claro cuando pasa esto:

```
ValueError: Continuous inference cannot handle discrete sample site 'toSample_5_0'. Consider enumerating that variable as documented in https://pyro.ai/examples/enumeration.html . If you are already enumerating, take care to hide this site when constructing an autoguide, e.g. guide = AutoNormal(poutine.block(model, hide=['toSample_5_0'])).
                Trace Shapes:     
                 Param Sites:     
                Sample Sites:     
            toTensor_6_0 dist | 89
                        value | 89
     toNaturalTensor_5_0 dist | 89
                        value | 89
     toNaturalTensor_4_0 dist | 89
                        value | 89
    toNatural0Tensor_3_0 dist | 89
                        value | 89
            toTensor_6_1 dist | 89
                        value | 89
toRealPositiveTensor_5_0 dist | 89
                        value | 89
            toTensor_8_0 dist | 89
                        value | 89
toRealPositiveTensor_7_0 dist | 89
                        value | 89
```

### ¿Soluciones?

- Encapsularlo en un `try-except` y si recibimos el error, asignarle un fitness extremo, indicando que dicho individuo
  no es válido.

[comment]: <> (    - **Contras**: Puede haber muchos individuos no válidos y ralentizar la evolución)

## 4. *NotImplementedError: Inhomogeneous total count not supported by `enumerate_support`*

Parece que no lo han implementado aún, pero tampoco sé cuando ocurre:

```
Traceback (most recent call last):
  ...
  File "/home/david/git/tfm/venv/lib/python3.9/site-packages/torch/distributions/binomial.py", line 118, in enumerate_support
    raise NotImplementedError("Inhomogeneous total count not supported by `enumerate_support`.")
NotImplementedError: Inhomogeneous total count not supported by `enumerate_support`.
```

### ¿Soluciones?

- Encapsularlo en un `try-except` y si recibimos el error, asignarle un fitness extremo, indicando que dicho individuo
  no es válido.

[comment]: <> (    - **Contras**: Puede haber muchos individuos no válidos y ralentizar la evolución)

## 5. *RuntimeError: The size of tensor a (2) must match the size of tensor b (89) at non-singleton dimension 0*

Error similar cuando `torch.pow(torch.Tensor([2, 0.5554]),torch.Tensor([0.9072, 2, 3]))`. Es una traza que saca la
función `safePow` en `grammar.py`

### ¿Soluciones?

- Encapsularlo en un `try-except` y si recibimos el error, asignarle un fitness extremo, indicando que dicho individuo
  no es válido.

[comment]: <> (    - **Contras**: Puede haber muchos individuos no válidos y ralentizar la evolución)

- Eliminar la operación de la gramática.

## 6. *RuntimeError: one of the variables needed for gradient computation has been modified*

```shell
RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.FloatTensor [1]], which is output 0 of PowBackward1, is at version 1; expected version 0 instead. Hint: the backtrace further above shows the operation that failed to compute its gradient. The variable in question was changed in there or anywhere later. Good luck!
Warmup:   0%|                                                                                    | 0/1250 [00:22, ?it/s]
```

# Parámetros de la inferencia

- `num_samples`?
- sampler type?