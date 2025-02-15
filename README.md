## **Introduction**  
This repository contains code for **simulations and visualizations** related to **Budget Proposal Aggregation**, a key area in participatory budgeting. The focus is on implementing and studying the **Independent Market Mechanism (IMM)**, a method used to fairly allocate public funds based on citizen votes.

For further reading, refer to:  
- [Wikipedia: Budget Proposal Aggregation](https://en.wikipedia.org/wiki/Budget-proposal_aggregation)  
- **Freeman et al. (2021)**: [https://arxiv.org/abs/1905.00457](https://arxiv.org/abs/1905.00457)  
- My own research (to be submitted 😉)
  

## Repository Structure

- `fyp_functions.py`: Contains the core functions to run my simulations and visualization.
- `main.ipynb`,`topic_1.ipynb`,`topic_2.ipynb`: Simluation and experiment playground
- `README.md`: Project overview and introduction.

## **Research Background: Participatory Budgeting**  
**Participatory Budgeting (PB)** is a democratic process where citizens vote on how public funds should be allocated in their communities (e.g., municipalities or neighborhoods). Unlike traditional representative democracy, PB enhances transparency, reduces corruption, and fosters inclusivity, particularly for marginalized groups. Initially introduced in Brazil (1989), PB has since expanded across Latin America, North America, and Europe.

### **Independent Market Mechanism (IMM)**  
#### **The Problem:**  
A well-known **phantom mechanism** (Moulin, 1980) **places `n+1` phantom votes among `n` actual votes** and selects the median as the allocation. While this ensures fairness and proportionality, **the total allocation may exceed 100% when there are more than two alternatives (`m > 2`)**.

#### **IMM Solution:**  
IMM strategically adjusts phantom votes to ensure that the sum of medians **always equals 1**. In Python terms, it **uses binary search to find an optimal scaling factor `t`** that balances the median-based allocation.

## **Code Usage**  
### **Running IMM on Random Preferences**  
```python  
from fyp_functions import fyp

P = fyp.generate_random_preferences(n=5, m=3, seed=2)  
allocation, mechanism_info, detailed_info = fyp.independent_market_mechanism(n, m, P)  
```

### **Visualizing Vote & Phantom Interactions**  
To better understand the impact of phantom votes, we can generate a visualization:
```python  
fyp.plot_vertical_allocation(P, show_phantoms=True)  
```
![pic](https://github.com/user-attachments/assets/0f6dcb22-a03e-4646-b61f-f3ad36fe09fa)  



