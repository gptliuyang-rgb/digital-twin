.PHONY: setup test lint check-spec build-assets eval-l0 eval-l1 eval-l2 ingest-official assemble view-industrial view-industrial-kinematic render-industrial-gif calibrate-synthetic eval-l2-overlay scan-gun-mesh phase1-baseline eval-l2-physics eval-l2-gains sim sim-quick frame-report

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

frame-report:
	$(PYTHON) -m eval.frame_alignment

view-industrial:
	$(PYTHON) scripts/view_industrial_twin.py --steps-per-phase 80 --substeps 24 --real-time

view-industrial-kinematic:
	$(PYTHON) scripts/view_industrial_twin.py --kinematic --steps-per-phase 100 --real-time

render-industrial-gif:
	$(PYTHON) scripts/render_industrial_gif.py --out artifacts/industrial_demo.gif --steps-per-phase 48 --substeps 12 --frame-stride 8 --fps 12

eval-l0:
	$(PYTHON) -m eval.l0_offline_replay --config eval/configs/l0_offline.yaml

eval-l1:
	$(PYTHON) -m eval.l1_kinematic --config eval/configs/l1_kinematic.yaml

eval-l2:
	$(PYTHON) -m eval.l2_mujoco_closedloop --config eval/configs/l2_mujoco.yaml

calibrate-synthetic:
	$(PYTHON) scripts/run_e1_e2_pipeline.py --both-skins --fit-tip-radius

calibrate-synthetic-l2:
	$(PYTHON) scripts/run_e1_e2_pipeline.py --both-skins --fit-tip-radius --with-l2

eval-l2-overlay:
	$(PYTHON) -m eval.l2_mujoco_closedloop \
		--spec hand/calibration/results/synthetic_batch_v1.0/generated/overlay.yaml \
		--out eval/report/generated/l2_overlay.json

scan-gun-mesh:
	$(PYTHON) -m assets.objects.scan_gun.generate

phase1-baseline:
	$(PYTHON) -m assets.dexhand2.build.pad_metrics

eval-l2-physics:
	$(PYTHON) -m eval.l2_mujoco_closedloop --physics-industrial --out eval/report/generated/l2_physics.json

eval-l2-gains:
	$(PYTHON) -m eval.l2_mujoco_closedloop --gain-scan-mode full --out eval/report/generated/l2_gains.json

sim:
	$(PYTHON) -m eval.run_sim_stack

sim-quick:
	$(PYTHON) -m eval.run_sim_stack --quick
