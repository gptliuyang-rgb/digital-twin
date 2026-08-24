.PHONY: setup test lint check-spec build-assets eval-l0 eval-l1 eval-l2 ingest-official assemble view-industrial render-industrial-gif calibrate-synthetic eval-l2-overlay scan-gun-mesh

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

build-assets:
	$(PYTHON) -m assets.dexhand2.build.gen_derived --side right
	$(PYTHON) -m assets.dexhand2.build.gen_derived --side left

assemble:
	$(PYTHON) -m assets.combined.assemble --check-compile

view-industrial:
	$(PYTHON) scripts/view_industrial_twin.py --steps-per-phase 100 --real-time

render-industrial-gif:
	$(PYTHON) scripts/render_industrial_gif.py --out artifacts/industrial_demo.gif --steps-per-phase 60 --frame-stride 20 --fps 12

eval-l0:
	$(PYTHON) -m eval.l0_offline_replay --config eval/configs/l0_offline.yaml

eval-l1:
	$(PYTHON) -m eval.l1_kinematic --config eval/configs/l1_kinematic.yaml

eval-l2:
	$(PYTHON) -m eval.l2_mujoco_closedloop --config eval/configs/l2_mujoco.yaml

calibrate-synthetic:
	$(PYTHON) scripts/run_e1_e2_pipeline.py

eval-l2-overlay:
	$(PYTHON) -m eval.l2_mujoco_closedloop \
		--spec hand/calibration/results/synthetic_batch_v1.0/generated/overlay.yaml \
		--out eval/report/generated/l2_overlay.json --skip-physics

scan-gun-mesh:
	$(PYTHON) -m assets.objects.scan_gun.generate
