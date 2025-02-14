# Constraint Acquisition with Query-Driven Refinement

This project implements a hybrid constraint acquisition system that integrates passive and active learning with a novel query-driven refinement step to mitigate overfitting. It focuses on learning global constraints (e.g., `AllDifferent`) from example solutions and refining the model by generating targeted violating assignments and querying an oracle.

## Features

- **Hybrid CA Framework:** Combines passive learning with active refinement.
- **Query-Driven Refinement:** Uses membership queries and Bayesian updates to remove overfitted constraints.
- **Global Constraint Support:** Focuses on the `AllDifferent` constraint.
- **Benchmarks:** Evaluated on Sudoku variants, Greater-Than Sudoku, JSudoku, and Exam Timetabling.

## How It Works

Our system follows a three-stage process that combines passive learning, query-driven refinement, and active learning to build an accurate constraint model.

### 1. Passive Learning

- **Candidate Extraction:**  
  The system begins by analyzing a set of example solutions to extract candidate global constraints, such as `AllDifferent`.  
- **Initial Model Formation:**  
  These candidates form an initial model that is consistent with the provided examples, though some constraints may be overfitted (i.e., they hold in the training data but do not generalize).

### 2. Query-Driven Refinement

- **Probability Estimation:**  
  A Random Forest classifier, trained on synthetic data from known CP models, assigns a prior probability to each candidate constraint. This probability indicates how likely it is that the constraint belongs to the true model.
  
- **Violation Query Generation:**  
  For each candidate `AllDifferent` constraint, the system generates a violating assignment by selecting a pair of variables within its scope and forcing them to take the same value—while ensuring that all other constraints remain satisfied.  
- **Oracle Interaction:**  
  The violating assignment is then submitted to an oracle.  
  - If the oracle accepts the assignment as valid, it indicates that the candidate constraint does not hold in the true model, and the constraint is removed.  
  - If the oracle rejects the assignment, the system uses Bayesian updating to increase its confidence that the candidate is valid.
  
- **Refinement Loop:**  
  This process repeats—generating new violating assignments and updating probabilities—until either the candidate is refuted or its confidence exceeds a predefined threshold.

### 3. Active Learning

- **Model Finalization:**  
  After refinement, the surviving global constraints are decomposed (e.g., `AllDifferent` is broken down into binary inequalities).  
- **Interactive Querying:**  
  An active learning algorithm then further refines the model by generating additional queries to learn any missing fixed-arity constraints.
  
- **Final Model:**  
  The combination of refined global constraints and actively learned fixed-arity constraints results in a complete and accurate constraint model.

This multi-stage approach ensures that overfitted constraints are identified and removed, leading to a model that generalizes well beyond the initial training examples.


## Acknowledgments

We would like to thank the developers and contributors of the following libraries for making this project possible:

- [CPMPy](https://github.com/CPMpy/cpmpy) – for providing a flexible and expressive constraint programming modeling framework.
- [OR-Tools](https://developers.google.com/optimization) – for its powerful solvers and optimization tools.
- [scikit-learn](https://scikit-learn.org/) – for the machine learning tools used in classifier training.

Additionally, we appreciate the support from the research community in the fields of constraint programming and constraint acquisition.
