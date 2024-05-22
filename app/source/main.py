import logging
from pathlib import Path
from source.config import load_config
import MEArec as mr
import MEAutility as mu
import numpy as np
import matplotlib.pyplot as plt

def setup_logging(config):
    logging_dir = Path(config["env"]["LOGGING_DIR"])
    # Create log file if it doesn't exist
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

        self.z_offset = self.config["simulation_params"]["between_session_z_offset"]

        self.output_dir = Path(self.config["env"]["OUTPUT_DIR"])

        self.tetrode_templates_path = self.output_dir / 'tetrode_templates.h5'

        self.neuronexus_templates_path = self.output_dir / 'neuronexus_templates.h5'

        self.redo_templates = bool(int(self.config["env"]["REDO_TEMPLATES"]))

        self.redo_recordings = bool(int(self.config["env"]["REDO_RECORDINGS"]))

        if self.redo_templates and self.tetrode_templates_path.exists():
            self.tetrode_templates_path.unlink()
        if self.redo_templates and self.neuronexus_templates_path.exists():
            self.neuronexus_templates_path.unlink()
        if self.redo_recordings:
            for file in self.output_dir.glob('*recording*.h5'):
                file.unlink()
        self.cell_models_dir = Path(self.config["env"]["CELL_MODELS_DIR"])
        self.n_cell_models = len([f for f in self.cell_models_dir.iterdir() if f.is_dir() and f.name != 'mods'])
        self.templates_params = self.config["templates_params"]
        self.recordings_params = self.config["recordings_params"]

    def run_simulation(self):
        print("Running tetrotrode simulation.")
        logging.info("Running tetrode simulation.")
        tempgen = self._handle_templates('tetrode')
        self._handle_recordings('tetrode', tempgen)
        print("Running Neuronexus-32 simulation.")
        logging.info("Running Neuronexus-32 simulation.")
        self._handle_templates('Neuronexus-32')
        self._handle_recordings('Neuronexus-32')
        print("Simulation complete.")
        logging.info("Pipeline complete")
        # modify API to return split_report
        logging.info("Pipeline complete")
        return {"message": "Simulation completed"}

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
        templates_params = self.templates_params.copy()
        templates_params["probe"] = probe
        templates_params["n"] = self.simulation_params["n_cells"] // self.n_cell_models
        tempgen = mr.gen_templates(
            cell_models_folder=self.cell_models_dir,
            params=templates_params,
            templates_tmp_folder=self.output_dir / f"{probe}_templates_tmp",
            delete_tmp=False,
            parallel=True,
            n_jobs=-1,
            verbose=True,
            recompile=False
            )
        print("Generated templates.")
        print(f"Saving templates to {self.tetrode_templates_path}.")
        print(tempgen.celltypes)
        if probe == 'tetrode':
            mr.save_template_generator(tempgen, filename=self.tetrode_templates_path)
        elif probe == 'Neuronexus-32':
            mr.save_template_generator(tempgen, filename=self.neuronexus_templates_path)
        logging.info("Generated and saved templates.")
        print("Generated and saved templates.")
        self._plot_locations(tempgen, probe_name=probe)
        print("Plotted locations.")
        return tempgen

    def _plot_locations(self, tempgen, probe_name):
        # plot locations
        prb = mu.return_mea(info=tempgen.info["electrodes"])
        ax = mu.plot_probe(prb)
        for loc in tempgen.locations[::5]:
            ax.plot([loc[0, 1], loc[-1, 1]], [loc[0, 2], loc[-1, 2]], alpha=0.7)
        ax.set_title(f"Locations of templates for {probe_name}")
        # save the ax plot
        plt.savefig(self.output_dir / f"{probe_name}_locations.png")



    def _handle_recordings(self, probe, tempgen):
        n_existing_recordings = len(list(self.output_dir.glob('{probe}_recording*.h5')))
        if n_existing_recordings == self.n_recordings:
            logging.info("Recordings already exist. Skipping generation.")
            print("Recordings already exist. Skipping generation.")
        elif n_existing_recordings == 0:
            self._generate_recordings(probe, tempgen)
        else:
            raise ValueError("Some recordings exist, but not all. Please delete all recordings or none.")

    def _generate_recordings(self, probe, tempgen):
        recordings_params = self.recordings_params.copy()
        if probe == 'tetrode':
            templates_path = self.tetrode_templates_path
            n_exc = 40
            n_inh = 10
        elif probe == 'Neuronexus-32':
            templates_path = self.neuronexus_templates_path
            n_exc = 320
            n_inh = 80
        else:
            raise ValueError(f"Unknown probe type: {probe}")
        recordings_params["spiketrains"]["n_exc"] = n_exc
        recordings_params["spiketrains"]["n_inh"] = n_inh
        recordings_params["recordings"]["n_drifting"] = n_exc + n_inh
        for i in range(self.n_recordings):
            if tempgen is None:
                recgen = mr.gen_recordings(
                    templates=templates_path,
                    params=recordings_params,
                    verbose=True,
                    n_jobs=-1
                    )
            else:
                recgen = mr.gen_recordings(
                    tempgen=tempgen,
                    params=recordings_params,
                    verbose=True,
                    n_jobs=-1,
                    drift_dicts=self._make_session_drift_dicts(probe, i)
                    )
            if probe == 'tetrode':
                rec_path = self.output_dir / f'tetrode_recording{i+1}.h5'
            elif probe == 'Neuronexus-32':
                rec_path = self.output_dir / f'neuronexus_recording{i+1}.h5'
            mr.save_recording_generator(recgen, filename=rec_path)
            logging.info(f"Generated and saved {self.n_recordings} recordings.")
            print(f"Generated and saved {self.n_recordings} recordings.")
            self._plot_cell_drifts(recgen, probe_name=probe, i=i)
            print(f"Plotted cell drifts for recording {i+1}.")

    def _plot_cell_drifts(self, recgen, probe_name, i):
        ax = mr.plot_cell_drifts(recgen)
        probe = recgen.info["probe"]
        ax.set_title(f"Cell drifts for {probe_name} recording {i+1}.")
        # save the ax plot
        plt.savefig(self.output_dir / f"{probe_name}_recording_{i}_cell_drifts.png")

    def _make_session_drift_dicts(self, probe, i):
        inter_session_drift_dict = self._make_intersession_drift_dict(i)
        if probe == 'tetrode':
            slow_drift = self.config["slow_rigid_drift_dict"].copy()
            fast_drift = self.config["fast_rigid_drift_dict"].copy()
        elif probe == 'Neuronexus-32':
            slow_drift = self.config["slow_flexible_drift_dict"].copy()
            fast_drift = self.config["fast_flexible_drift_dict"].copy()
        return [inter_session_drift_dict, slow_drift, fast_drift]

    def _make_intersession_drift_dict(self, i):
        inter_session_drift_dict = self.config["base_signal_drift_dict"].copy()
        duration = self.recordings_params["spiketrains"]["duration"]
        drift_times = np.arange(0, duration, 0.1)
        depth = self.z_offset * i
        # drift vector is a vector of the same shape as drift_times with the depth value repeated
        drift_vector = np.full_like(drift_times, depth)
        #drift_factors = np.full_like(drift_times, 1.0)
        inter_session_drift_dict["external_drift_times"] = drift_times
        inter_session_drift_dict["external_drift_vector_um"] = drift_vector
        inter_session_drift_dict["external_drift_factors"] = None
        return inter_session_drift_dict


def main():
    config_dict = load_config()
    setup_logging(config_dict)
    logging.info("===== Starting simulation. =====")
    simulator = Simulator(config=config_dict)
    simulator.run_simulation()
    logging.info("===== Simulation finished. =====")

if __name__ == "__main__":
    main()