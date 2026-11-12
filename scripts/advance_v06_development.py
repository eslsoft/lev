"""Execute the registered v06 development stages once after successful training."""
import fcntl
import json
from pathlib import Path
import subprocess
import time

from lev.io import digest, read, write

ROOT = Path('runs/lev-joint005-v06-cycle')


def main():
    import modal
    with (ROOT / 'development-controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = read(ROOT / 'post-training-plan.json')
        if plan['registration_sha256'] != digest(ROOT / 'registration.json'):
            raise ValueError('Registration changed')
        registration = read(ROOT / 'registration.json')
        if registration['dataset_sha256'] != digest('artifacts/en-joint-005/manifest.json') or registration['config_sha256'] != digest('configs/lev-4b-v06.toml'):
            raise ValueError('Registered inputs changed')
        volume = modal.Volume.from_name('lev-experiments')
        print('Waiting for existing v06 training; no training submission', flush=True)
        while True:
            try:
                status = json.loads(b''.join(volume.read_file('lev-joint005-v06-train-01/modal-status.json')))
            except FileNotFoundError:
                time.sleep(30)
                continue
            if status['status'] != 'succeeded':
                raise RuntimeError(f'Training did not succeed: {status}')
            break
        for stage in plan['stages']:
            name = stage['name']
            started, finished = ROOT / (name + '.started.json'), ROOT / (name + '.finished.json')
            if started.exists() or finished.exists():
                raise RuntimeError(f'Stage already recorded; inspect before resuming: {name}')
            write(started, stage)
            print('Starting ' + name, flush=True)
            with (ROOT / (name + '.log')).open('x') as log:
                process = subprocess.run(stage['command'], stdout=log, stderr=subprocess.STDOUT)
            write(finished, dict(returncode=process.returncode))
            if process.returncode:
                raise RuntimeError(f'Stage failed: {name}')
        write(ROOT / 'development-execution.json', dict(status='evaluations_downloaded_pending_gate_verification', final_goal_complete=False))


if __name__ == '__main__':
    main()
