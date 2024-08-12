# Infant behaviour using agent based modelling

## Simulation parameters

The simulation is based on a couple of parameters, defined for either parent, infant, or independently from any of the agents.

### Infant rules/ traits

* precision - determines how much the infant is focused on playing with a certain toy - ranges from being completely distracted to total focus on a current toy
* coordination - how likely the infant is to perform a coordinated action with an object - i.e. throw it towards its guardian rather than throw it in an undefined direction
* exploration - describes how much the infant prefers playing with toys it has already played with (value of 0) or the toys it has played the least with (100). The value of 50 represents no preference for any of the toys

### Parent traits
* responsiveness - how likely the parent is to respond when the infant throws a toy (no matter if the infant throws the toy toward the parent, or in any other direction)
* relevance - determines if the parent's action is relevant to the infant's behavior. The higher the value, the more likely the parent is to pick up a toy, which has just been played with by the infant.

### Other parameters
* number of toys on the board


## How to Run

Launch the simulation:
```console
mesa runserver
```

The browser should automatically open at [http://127.0.0.1:8521/](http://127.0.0.1:8521/).

