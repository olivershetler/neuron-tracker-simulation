import logging
from pathlib import Path
from source.config import load_config
import MEArec as mr
import numpy as np

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
    def __init__(self, config, redo_templates=False, redo_recordings=False):
        self.config = config
        self.n_recordings = self.config["session_sequence_params"]["n_recordings"]
        self.z_offset = self.config["session_sequence_params"]["between_session_z_offset"]
        self.output_dir = Path(self.config["env"]["OUTPUT_DIR"])
        self.tetrode_templates_path = self.output_dir / 'tetrode_templates.h5'
        self.neuronexus_templates_path = self.output_dir / 'neuronexus_templates.h5'
        if redo_templates and self.tetrode_templates_path.exists():
            self.tetrode_templates_path.unlink()
        if redo_templates and self.neuronexus_templates_path.exists():
            self.neuronexus_templates_path.unlink()
        if redo_recordings:
            for file in self.output_dir.glob('*recording*.h5'):
                file.unlink()
        self.cell_models_dir = Path(self.config["env"]["CELL_MODELS_DIR"])
        self.template_params = self.config["template_params"]
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
        template_params = self.template_params.copy()
        template_params["probe"] = probe
        tempgen = mr.gen_templates(
            cell_models_folder=self.cell_models_dir,
            params=template_params,
            n_jobs=-1,
            verbose=True,
            recompile=True
            )
        if probe == 'tetrode':
            mr.save_template_generator(tempgen, filename=self.tetrode_templates_path)
        elif probe == 'Neuronexus-32':
            mr.save_template_generator(tempgen, filename=self.neuronexus_templates_path)
        logging.info("Generated and saved templates.")
        print("Generated and saved templates.")
        return tempgen

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
        if probe == 'tetrode':
            templates_path = self.tetrode_templates_path
        elif probe == 'Neuronexus-32':
            templates_path = self.neuronexus_templates_path
        else:
            raise ValueError(f"Unknown probe type: {probe}")
        if tempgen is None:
            tempgen = mr.load_templates(templates_path)
        for i in range(self.n_recordings):
            recgen = mr.gen_recordings(
                tempgen=tempgen,
                params=self.recordings_params,
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
        inter_session_drift_dict["external_drift_times"] = drift_times
        inter_session_drift_dict["external_drift_vector_um"] = drift_vector
        return inter_session_drift_dict

def main():
    config_dict = load_config()
    setup_logging(config_dict)
    logging.info("===== Starting simulation. =====")
    simulator = Simulator(config=config_dict, redo_templates=True, redo_recordings=True)
    simulator.run_simulation()
    logging.info("===== Simulation finished. =====")

if __name__ == "__main__":
    main()