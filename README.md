# Master Thesis

*Evolutionary Computation in Hierarchical Model Discovery*

David Revillas, 2022

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
