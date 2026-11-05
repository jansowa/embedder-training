import importlib.util
from pathlib import Path
from types import SimpleNamespace

path=Path(__file__).resolve().parents[1]/'experiments/polish-splade-2026-10/queues/run_paired_night3.py'
spec=importlib.util.spec_from_file_location('night3_runner',path)
runner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_gm_requires_both_densities_and_complete_scores():
    assert runner.gate({'dims':400,'tiny10':58},{'dims':390,'tiny10':55}).startswith('PASS')
    assert 'nieporównywalne' in runner.gate({'dims':236,'tiny10':59},{'dims':390,'tiny10':55})
    assert 'GM niepoliczony' in runner.gate({'dims':400},{'dims':390,'tiny10':55})
    assert runner.gate({'dims':400,'tiny10':56},{'dims':390,'tiny10':55.5}).startswith('distilroberta')
    assert runner.gate({'dims':400,'tiny10':56},{'dims':390,'tiny10':55}).startswith('MAYBE')


def test_selected_checkpoint_forces_save_without_extra_periodic_save():
    from training.checkpoints import StepCheckpointCallback
    callback=StepCheckpointCallback(output_dir='unused',steps=[2150,4306])
    for step,expected in [(2150,True),(4300,False),(4306,True)]:
        control=SimpleNamespace(should_save=False)
        callback.on_step_end(None,SimpleNamespace(global_step=step),control)
        assert control.should_save==expected
