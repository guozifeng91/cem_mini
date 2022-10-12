# Combinatorial Equilibrium Modeling (CEM) solver mini

cem_mini is a python implementation of the CEM method originally proposed by Dr. Ohlbrock, P. O., ETH Zurich

cem_mini aims to minimize the number of dependent libraries<br>
cem_mini attempts to stay with function-oriented-programming style and simplifies the data structure

cem_mini is implemented by Z. Guo, following the chapter 7 of Dr. Ohlbrock 's PhD thesis [Combinatorial Equilibrium Modelling](https://www.research-collection.ethz.ch/handle/20.500.11850/478732)<br>

the implementation is validated using [CEM Grasshopper](https://github.com/OleOhlbrock/CEM) developed by Ohlbrock, Patrick Ole and D'Acunto, Pierluigi, and [compas_cem](https://github.com/arpastrana/compas_cem) developed by Rafael Pastrana

## dependent
[numpy](https://numpy.org/)

[matplotlib](https://matplotlib.org/)

## examples

some basic examples [source code](src/cem_mini/cem_examples.py) and [jupyter notebook](src/examples.ipynb )

examples on making [bridges](src/example_bridges.ipynb)

![Alt text](quick_start.png?raw=true "quick start")

![Alt text](braced_tower.png?raw=true "braced tower")

![Alt text](bridge.png?raw=true "bridge 2d")

![Alt text](random-tower.png?raw=true "random tower")

![Alt text](bridge2.png?raw=true "bridge 3d")
