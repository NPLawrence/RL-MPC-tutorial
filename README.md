# RL-MPC-tutorial

 
This repo contains the slides and supporting code for my tutorial on [reinforcement learning and MPC at Upper Bound 2024](https://www.upperbound.ai/speakers/SPEWNJXVTAX).

To view the slides, go to this page's [site](https://nplawrence.com/RL-MPC-tutorial/).

---

At some point, I will also post a link to a companion paper I am working on. You can follow updates here or follow me on [Google Scholar](https://scholar.google.com/citations?user=Fe9p7QoAAAAJ&hl).

---

## Code

Code is contained [here](./code). It's a mix of Python and Julia. 

To run the Python code in [RL_MPC](./code/RL_MPC) or [double_inverted_pendulum](./code/double_inverted_pendulum), clone this repo, create/activate a virtual environment, `cd` to the `code` directory then install from `requirements.txt`:
```
pip install -r requirements.txt
```

To run the Julia code in [PID_LQR](./code/PID_LQR), clone this repo, navigate to `PID_LQR` in the Julia REPL then type the commands:
```
]
activate .
instantiate
```
