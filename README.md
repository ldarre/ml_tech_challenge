# Mercado Libre Tech Challenge

Notebooks (and accompanying files if needed) with the solutions for the addressed parts of the challenge can be found in the folder:
- Part-1: EDA for lightning deals (`eda.ipynb`)
- Part-3: Products similarity (`similarity.ipynb`, `output.csv`)
- Part-4: Time series forecasting (`forecasting.ipynb`).

In the case of trained models of Part-4 they are not uploaded to the repo, but can be sent upon request.


# Notebooks use

To execute the notebooks, first setup the environment. I would recommend using `Poetry` with the provided `poetry.lock` file or using the `pyproject.toml`. 

If using Poetry:
1. Install poetry following the [docs](https://python-poetry.org/docs/#installation)
2. Configure so that virtual envs are created inside each project

```bash
poetry config virtualenvs.in-project true
```

3. Create env
```bash
poetry shell
```

4. Install dependecies
```bash
poetry install
```

5. Run the notebooks using `jupyter-notebook` or `vscode`




