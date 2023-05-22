# Toddler behaviour using agent based modelling

## Installation and running

Clone the repo:
```
git clone https://github.com/roznawsk/toddler-abm
cd toddler-abm
```

Go to the continous space simulation:
```
cd cont_model
```

Run mesa server:
```
mesa runserver
```

A browser window with the simulation should open automatically.

## Toddler rules/ traits


### Autonomus traits - indepentend from the guardian
* precision - determines how much the toddler is focused on a goal - ranges from being completely distracted and unorganized to total focus on a certain task
* perception - describes toddler's ability to notice a toy
* coordination - how likely the toddler is to perform a coordinated action with an object - e.g. gently put it down or throw it in a determined direction, rather than dropping it or throwing carelessly

### Reliant traits - relating to the guardian
* dependence - how much affirmation from the guardian the toddler requires
* attention seeking - how much does the toddler struggle to gain parent's attention


## Parent traits
* guidance - how close the parent tries to stay to the toddler
* responsiveness - how likely the parent is to respond to the toddlers actions
* relevancy - how pertient are the parents responses