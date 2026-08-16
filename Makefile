.PHONY: setup test lint check-spec build-assets eval-l0 eval-l1 eval-l2 ingest-official check-drift eval-qr ingest-t800

PYTHON ?= python3

setup:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest tests

lint:
	$(PYTHON) -m ruff check interface hand runtime vla sim eval assets tests scripts

check-spec:
	$(PYTHON) -c "from interface.schema import validate_repo_spec; validate_repo_spec()"

ingest-official:
	$(PYTHON) -m assets.dexhand2.build.ingest_official

ingest-t800:
	$(PYTHON) -m assets.engineai.build.ingest_t800 --write

build-assets:
	$(PYTHON) -m assets.dexhand2.build.gen_derived --side both --simplified

check-drift:
	$(PYTHON) scripts/check_upstream_drift.py

eval-l0:
	$(PYTHON) -m eval.l0_offline_replay --config eval/configs/l0_offline.yaml

eval-l1:
	$(PYTHON) -m eval.l1_kinematic --config eval/configs/l1_kinematic.yaml

eval-l2:
	$(PYTHON) -m eval.l2_mujoco_closedloop --config eval/configs/l2_mujoco.yaml

eval-qr:
	$(PYTHON) -m eval.qr_envelope
