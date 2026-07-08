# Reproducibility: raw -> global-grain data -> features -> models -> figures.
# Stub targets; wire to the ported modules as the refactor lands.

PY ?= python

.PHONY: all data features models figures test clean

all: data features models figures

## build the global-grain data (grid_global, crops_global, surplus_global, weather_global)
data:
	$(PY) -m src.build.make_data

## (placeholder) build feature frames from the global tables
features:
	@echo "TODO: src/features pipeline"

## (placeholder) train + tune reg/clf models -> artifacts/, results/model_comparison.csv
models:
	@echo "TODO: src/models train + tune"

## (placeholder) regenerate final figures/tables -> results/
figures:
	@echo "TODO: notebooks/final_results or a figures script"

## run the test suite (incl. the sustag-vs-sustag2 parity guardrail)
test:
	$(PY) -m pytest tests/ -q

clean:
	@echo "TODO: remove derived artifacts under src/data/{interim,processed,cache}"
