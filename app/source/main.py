import logging
from pathlib import Path
from source.config import load_config
import MEArec as mr
import MEAutility as mu
import numpy as np
import matplotlib.pyplot as plt
import os

def setup_logging(config):
    logging_dir = Path(config["env"]["LOGGING_DIR"])
    for parent in logging_dir.parents:
        if not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)
    logging_dir.mkdir(parents=True, exist_ok=True)
    log_path = logging_dir / "simulation.log"
    log_path.touch(exist_ok=True)
    # Set up logging
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
        )

class Simulator:
    def __init__(self, config):
        self.config = config
        self.n_recordings = self.config["simulation_params"]["n_recordings"]

        self.output_dir = Path(self.config["env"]["OUTPUT_DIR"])

        self.tetrode_templates_path = self.output_dir / 'tetrode_templates.h5'

        self.neuronexus_templates_path = self.output_dir / 'neuronexus_templates.h5'

        self.redo_templates = bool(int(self.config["env"]["REDO_TEMPLATES"]))

        self.redo_recordings = bool(int(self.config["env"]["REDO_RECORDINGS"]))

        if self.redo_templates and self.tetrode_templates_path.exists():
            self.tetrode_templates_path.unlink()
        if self.redo_templates and self.neuronexus_templates_path.exists():
            self.neuronexus_templates_path.unlink()
            for file in self.output_dir.glob('*_locations.png'):
                file.unlink()
        if self.redo_recordings:
            for file in self.output_dir.glob('*recording*.h5'):
                file.unlink()
            for file in self.output_dir.glob('*_cell_drifts.png'):
                file.unlink()
        self.cell_models_dir = Path(self.config["env"]["CELL_MODELS_DIR"])
        self.n_cell_models = len([f for f in self.cell_models_dir.iterdir() if f.is_dir() and f.name != 'mods'])
        print(f"Found {self.n_cell_models} cell models.")

        # Adjust this value if you want to reserve some CPUs for other processes
        all_cpus = os.cpu_count()
        CPU_USAGE_FACTOR = 0

        self.available_cpus = max(1, int(all_cpus * CPU_USAGE_FACTOR))
        print(f"Using {self.available_cpus} CPUs for simulation.")
        self.region_num = int(self.config["env"]["REGION_NUM"])

    def run_simulation(self, probe):
        if probe not in ['tetrode', 'Neuronexus-32']:
            raise ValueError("Probe must be 'tetrode' or 'Neuronexus-32'.")
        print(f"Running {probe} simulation.")
        logging.info(f"Running {probe} simulation.")
        tempgen = self._handle_templates(probe)
        self._handle_recordings(probe, tempgen)
        print(f"{probe} simulation complete.")
        logging.info(f"{probe} simulation complete.")

    def _handle_templates(self, probe):
        if probe == 'tetrode' and not self.tetrode_templates_path.exists():
            tempgen = self._generate_templates(probe)
        elif probe == 'Neuronexus-32' and not self.neuronexus_templates_path.exists():
            tempgen = self._generate_templates(probe)
        elif probe == 'tetrode' or probe == 'Neuronexus-32':
            logging.info("Templates already exist. Will load from file.")
            print("Templates already exist. Will load from file.")
            return None
        else:
            raise ValueError("Probe must be 'tetrode' or 'Neuronexus-32'.")
        return tempgen

    def _generate_templates(self, probe):
        if probe == 'tetrode':
            templates_params = self.config["tetrode_templates_params"].copy()
        elif probe == 'Neuronexus-32':
            templates_params = self.config["neuronexus_templates_params"].copy()
        tempgen = mr.gen_templates(
            cell_models_folder=self.cell_models_dir,
            params=templates_params,
            templates_tmp_folder=self.output_dir / f"{probe}_templates_tmp",
            delete_tmp=False,
            parallel=True,
            n_jobs=8,#self.available_cpus,
            verbose=True,
            recompile=False
        )
        print("Generated templates.")
        if probe == 'tetrode':
            mr.save_template_generator(tempgen, filename=self.tetrode_templates_path)
            print("Saved templates locally.")
        elif probe == 'Neuronexus-32':
            mr.save_template_generator(tempgen, filename=self.neuronexus_templates_path)
            print("Saved templates locally.")
        logging.info("Generated and saved templates.")
        print("Generated and saved templates.")
        self._plot_locations(tempgen, probe_name=probe)
        print("Plotted locations.")
        return tempgen

    def _plot_locations(self, tempgen, probe_name):
        prb = mu.return_mea(info=tempgen.info["electrodes"])
        ax = mu.plot_probe(prb)
        for loc in tempgen.locations[::5]:
            ax.plot([loc[0, 1], loc[-1, 1]], [loc[0, 2], loc[-1, 2]], alpha=0.7)
        ax.set_title(f"Locations of templates for {probe_name}")
        fpath = self.output_dir / f"{probe_name}_locations.png"
        plt.savefig(fpath)
        print(f"Saved locations plot for {probe_name} to GCS.")

    def _handle_recordings(self, probe, tempgen):
        n_existing_recordings = len(list(self.output_dir.glob(f"{probe}_recording*.h5")))
        if n_existing_recordings == self.n_recordings:
            logging.info("Recordings already exist. Skipping generation.")
            print("Recordings already exist. Skipping generation.")
        elif n_existing_recordings == 0:
            self._generate_recordings(probe, tempgen)
        else:
            raise ValueError("Some recordings exist, but not all. Please delete all recordings or none.")

    def _generate_recordings(self, probe, tempgen):
        if probe == 'tetrode':
            templates_path = self.tetrode_templates_path
            recordings_params = self.config["tetrode_recordings_params"].copy()
        elif probe == 'Neuronexus-32':
            templates_path = self.neuronexus_templates_path
            recordings_params = self.config["neuronexus_recordings_params"].copy()
        else:
            raise ValueError(f"Unknown probe type: {probe}")
        cur_depth = self._init_cur_depth(probe)
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29] # handles up to 10 recordings
        for i in range(self.n_recordings):
            np.random.seed(primes[i]**self.region_num) # each seed is unique by the fundamental theorem of arithmetic
            if tempgen is None:
                recgen = mr.gen_recordings(
                    templates=templates_path,
                    params=recordings_params,
                    verbose=True,
                    n_jobs=self.available_cpus,
                    drift_dicts=self._make_session_drift_dicts(probe, cur_depth=cur_depth)
                )
            else:
                recgen = mr.gen_recordings(
                    tempgen=tempgen,
                    params=recordings_params,
                    verbose=True,
                    n_jobs=self.available_cpus,
                    drift_dicts=self._make_session_drift_dicts(probe, cur_depth=cur_depth)
                )
            if probe == 'tetrode':
                rec_path = self.output_dir / f'tetrode_recording_{i+1}.h5'
            elif probe == 'Neuronexus-32':
                rec_path = self.output_dir / f'neuronexus_recording_{i+1}.h5'
            mr.save_recording_generator(recgen, filename=rec_path)
            print(f"Saved recording {i+1} to {rec_path}.")
            self._plot_cell_drifts(recgen, probe_name=probe, i=i)
            cur_depth = self._update_intersession_depth(cur_depth, probe)

    def _update_intersession_depth(self, cur_depth, probe):
        if probe == 'Neuronexus-32':
            range = self.config["neuronexus_templates_params"]["drift_zlim"][0]
            lower_bound = -range//2
            upper_bound = range//2
            jump = np.random.normal(0, 10)
            new_depth = cur_depth + jump
            if new_depth > upper_bound:
                new_depth = cur_depth - abs(jump)
            elif new_depth < lower_bound:
                new_depth = cur_depth + abs(jump)
            return new_depth
        elif probe == 'tetrode':
            return cur_depth

    def _init_cur_depth(self, probe):
        if probe == 'Neuronexus-32':
            return 0
        elif probe == 'tetrode':
            return -10
        else:
            raise ValueError(f"Unknown probe type: {probe}")

    def _plot_cell_drifts(self, recgen, probe_name, i):
        ax = mr.plot_cell_drifts(recgen)
        ax.set_title(f"Cell drifts for {probe_name} recording {i+1}.")
        savepath = self.output_dir / f"{probe_name}_recording_{i+1}_cell_drifts.png"
        plt.savefig(savepath)
        print(f"Saved cell drifts plot for {probe_name} recording {i+1} to GCS.")

    def _make_session_drift_dicts(self, probe, cur_depth):
        inter_session_drift_dict = self._make_intersession_drift_dict(probe, cur_depth=cur_depth)
        slow_drift = self._make_slow_external_drift_dict(probe, cur_depth=cur_depth)
        fast_drift = self._make_fast_external_drift_dict(probe, cur_depth=cur_depth)
        return [inter_session_drift_dict, slow_drift, fast_drift]

    def _make_intersession_drift_dict(self, probe, cur_depth):
        if probe == 'Neuronexus-32':
            duration = self.config["neuronexus_recordings_params"]["spiketrains"]["duration"]
            range = self.config["neuronexus_templates_params"]["drift_zlim"][0]
            lower_bound = -range//2
            upper_bound = range//2
        else:
            raise ValueError(f"Only Neuronexus-32 probe is supported for inter-session drift right now. Tetrode is specifically not supported because it cannot detect far drifting neurons. Got {probe}.")
        inter_session_drift_dict = self.config["base_signal_drift_dict"].copy()
        drift_fs = 100
        drift_step = 1 / drift_fs
        drift_times = np.arange(0, duration + drift_step, drift_step)
        assert cur_depth <= upper_bound, f"Depth {cur_depth} is greater than upper bound {upper_bound}."
        assert cur_depth >= lower_bound, f"Depth {cur_depth} is less than lower bound {lower_bound}."
        drift_vector = np.full_like(drift_times, cur_depth)
        inter_session_drift_dict["external_drift_times"] = drift_times
        inter_session_drift_dict["external_drift_vector_um"] = drift_vector
        inter_session_drift_dict["external_drift_factors"] = None
        return inter_session_drift_dict

    def _make_fast_external_drift_dict(self, probe, cur_depth):
        if probe == 'Neuronexus-32':
            duration = self.config["neuronexus_recordings_params"]["spiketrains"]["duration"]
        elif probe == 'tetrode':
            duration = self.config["tetrode_recordings_params"]["spiketrains"]["duration"]
        fast_normal_noise = self.config["base_signal_drift_dict"].copy()
        drift_fs = 100
        drift_step = 1 / drift_fs
        drift_times = np.arange(0, duration + drift_step, drift_step)
        drift_vector_um = np.random.normal(0, 1, size=len(drift_times))
        fast_normal_noise["external_drift_times"] = drift_times
        fast_normal_noise["external_drift_vector_um"] = drift_vector_um
        fast_normal_noise["external_drift_factors"] = None
        return fast_normal_noise

    def _make_slow_external_drift_dict(self, probe, cur_depth):
        if probe == 'Neuronexus-32':
            duration = self.config["neuronexus_recordings_params"]["spiketrains"]["duration"]
        slow_walk_drift = self.config["base_signal_drift_dict"].copy()
        drift_fs = 100
        drift_step = 1 / drift_fs
        drift_times = np.arange(0, duration + drift_step, drift_step)
        drift_vector = np.zeros_like(drift_times)
        max_drift = 10
        for i in range(1, len(drift_times)):
            jump = np.random.normal(0, 0.1)
            drift_vector[i] = drift_vector[i - 1] + jump
            if drift_vector[i] >= max_drift or drift_vector[i - 1] >= max_drift:
                drift_vector[i] = drift_vector[i - 1] - abs(jump)
            elif drift_vector[i] <= -max_drift or drift_vector[i - 1] <= -max_drift:
                drift_vector[i] = drift_vector[i - 1] + abs(jump)
        slow_walk_drift["external_drift_times"] = drift_times
        slow_walk_drift["external_drift_vector_um"] = drift_vector
        slow_walk_drift["external_drift_factors"] = None
        return slow_walk_drift

def main():
    config_dict = load_config()
    setup_logging(config_dict)
    logging.info("===== Starting simulation. =====")
    simulator = Simulator(config=config_dict)
    for probe in ['Neuronexus-32']:#, 'tetrode']:
        simulator.run_simulation(probe)
    logging.info("===== Simulation finished. =====")

if __name__ == "__main__":
    main()