.PHONY: setup test lint check-spec build-assets eval-l0 eval-l1 eval-l2 ingest-official check-drift eval-qr ingest-t800 eval-report weld-recipe sonic-status gmr-export gmr-tpose usd-pads eval-l2-priv eval-l3-priv eval-gain-scan eval-l1a eval-l1a-spring eval-l1a-stream eval-l1a-operator eval-l1a-gather eval-l1a-encoder eval-l1a-planner-onnx eval-l1a-planner-blend eval-l1a-idle-readapt eval-l1a-playback eval-l0-diagnose eval-l1-case-a extract-kinematics ppo-status ppo-train eval-l2-sim2sim eval-l2-physics-sim2sim eval-l2-freebase-stand eval-l2-freebase-push eval-l3-isaac-bind

PYTHON ?= python3

setup:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest tests

lint:
	$(PYTHON) -m ruff check interface hand runtime vla sim eval assets wbc tests scripts

check-spec:
	$(PYTHON) -c "from interface.schema import validate_repo_spec; validate_repo_spec()"

ingest-official:
	$(PYTHON) -m assets.dexhand2.build.ingest_official

ingest-t800:
	$(PYTHON) -m assets.engineai.build.ingest_t800 --write

extract-kinematics:
	$(PYTHON) -m assets.engineai.build.extract_kinematics --write

build-assets:
	$(PYTHON) -m assets.dexhand2.build.gen_derived --side both --simplified

check-drift:
	$(PYTHON) scripts/check_upstream_drift.py

eval-l0:
	$(PYTHON) -m eval.l0_offline_replay --config eval/configs/l0_offline.yaml

eval-l0-diagnose:
	$(PYTHON) -m eval.l0_ckpt_diagnose

eval-l1:
	$(PYTHON) -m eval.l1_kinematic --config eval/configs/l1_kinematic.yaml

eval-l1-case-a:
	$(PYTHON) -m eval.l1_case_a --apply-fk

eval-l2:
	$(PYTHON) -m eval.l2_mujoco_closedloop --config eval/configs/l2_mujoco.yaml

eval-qr:
	$(PYTHON) -m eval.qr_envelope

eval-report:
	$(PYTHON) -m eval.report.generate

weld-recipe:
	$(PYTHON) -m assets.combined.assemble || true

sonic-status:
	$(PYTHON) -c "from wbc.retarget import status_report; import json; print(json.dumps(status_report(), indent=2, default=str))"

gmr-tpose:
	$(PYTHON) -m wbc.gmr.tpose --write
	$(PYTHON) -m wbc.gmr.export

gmr-export:
	$(PYTHON) -m wbc.gmr.export

usd-pads:
	$(PYTHON) -m assets.dexhand2.build.gen_usd_pads --side right
	$(PYTHON) -c "from assets.dexhand2.build.gen_usd_pads import generate; generate('left')"

eval-l2-priv:
	$(PYTHON) -m eval.l2_privileged

eval-l3-priv:
	$(PYTHON) -m eval.l3_isaac_privileged

eval-gain-scan:
	$(PYTHON) -m eval.gain_scan

eval-l1a:
	$(PYTHON) -c "from wbc.planner import KinematicPlanner; from interface.schema import CommandVector; p=KinematicPlanner(); r=p.plan([CommandVector.zeros(), CommandVector.zeros()]); print(r.n_steps, r.rate_hz, r.horizon_s)"

eval-l1a-spring:
	$(PYTHON) -m eval.l1a_spring

eval-l1a-stream:
	$(PYTHON) -m eval.l1a_stream

eval-l1a-operator:
	$(PYTHON) -m eval.l1a_operator

eval-l1a-gather:
	$(PYTHON) -m eval.l1a_gather

eval-l1a-encoder:
	$(PYTHON) -m eval.l1a_encoder

eval-l1a-planner-onnx:
	$(PYTHON) -m eval.l1a_planner_onnx

eval-l1a-planner-blend:
	$(PYTHON) -m eval.l1a_planner_blend

eval-l1a-idle-readapt:
	$(PYTHON) -m eval.l1a_idle_readapt

eval-l1a-playback:
	$(PYTHON) -m eval.l1a_playback

ppo-status:
	$(PYTHON) -c "from wbc.ppo.recipe import status_report; import json; print(json.dumps(status_report(), indent=2, default=str))"

ppo-train:
	$(PYTHON) -c "from wbc.ppo.recipe import refuse_ppo_launch; refuse_ppo_launch()"

eval-l2-sim2sim:
	$(PYTHON) -m eval.l2_sim2sim

eval-l2-physics-sim2sim:
	$(PYTHON) -m eval.l2_physics_sim2sim

eval-l2-freebase-stand:
	$(PYTHON) -m eval.l2_freebase_stand

eval-l2-freebase-push:
	$(PYTHON) -m eval.l2_freebase_push

eval-l3-isaac-bind:
	$(PYTHON) -m eval.l3_isaac_bind
