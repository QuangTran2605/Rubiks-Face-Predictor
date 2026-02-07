# Rubik's Cube Last Face Predictor
We are interested in whether we can, given 5 faces of a Rubik's Cube, predict the last face's pattern, using a ML model.  

## Description
We'll start by feeding a set of images (2-5) to the model, which captures the patterns on 5 faces of the cube sitting on a table. Then the task is to figure out the exact pattern on the last face, which is hidden from view. 

## Data
We use random cube states created through 20 shuffling steps, with a total 10000 examples, splitted into train, validation, and test sets.  

## Model
TBD

## Status
02/03/26: Cube simulator and data generator created. 
02/07/26: Cube simulator and data generator debugged to filter impossible cube states and illegal moves. ExactSolver created to establish a deterministic algorithm baseline, which consists of center, corner, and edge solvers, then combines them and returns a set of possible solutions for the hidden face. Visualization of solutions created for the ExactSolver.  