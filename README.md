## Sep 27

- Starting with JPO component, I defined and developed a standardized template of DL workload deployment and two small scripts that extract the new-coming jobs based on this standard.
    - Developed both in bash and python.
    - Hyperparameters and constants are defined in a file named as `config.ini` next to the `app.py`.
    - Scripts check these two files and report these items:
        - `config.ini` syntax and data type validity.
        - `app.py` usage of `config.ini`.
        - DL main characteristics:
            - Framework
            - Dataset
            - Model

## To-Do

- We can use this to implement the SAC part:
    - https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py