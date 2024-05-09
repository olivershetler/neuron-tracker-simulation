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
        self.templates_path = self.output_dir / 'templates.h5'
        if redo_templates and self.templates_path.exists():
            self.templates_path.unlink()
        if redo_recordings:
            for file in self.output_dir.glob('recording*.h5'):
                file.unlink()
        self.cell_models_dir = Path(self.config["env"]["CELL_MODELS_DIR"])
        self.template_params = self.config["template_params"]
        self.recordings_params = self.config["recordings_params"]

    def run_simulation(self):
        self._handle_templates()
        self._handle_recordings()
        # modify API to return split_report
        logging.info("Pipeline complete")
        return {"message": "Simulation completed"}

    def _handle_templates(self):
        if not self.templates_path.exists():
            self._generate_templates()
        else:
            logging.info("Templates already exist. Loading from file.")
            self._load_templates()

    def _generate_templates(self):
        tempgen = mr.gen_templates(
            cell_models_folder=self.cell_models_dir,
            params=self.template_params,
            n_jobs=13
            )
        mr.save_template_generator(tempgen, filename=self.templates_path)
        logging.info("Generated and saved templates.")
        self.tempgen = tempgen

    def _load_templates(self):
        tempgen = mr.load_templates(self.templates_path)
        logging.info(f"Loaded pre-saved templates from {self.templates_path} .")
        self.tempgen = tempgen

    def _handle_recordings(self):
        n_existing_recordings = len(list(self.output_dir.glob('recording*.h5')))
        if n_existing_recordings == self.n_recordings:
            logging.info("Recordings already exist. Skipping generation.")
            self._load_recordings()
        elif n_existing_recordings == 0:
            self._generate_recordings()
        else:
            raise ValueError("Some recordings exist, but not all. Please delete all recordings or none.")

    def _generate_recordings(self):
        for i in range(self.n_recordings):
            recgen = mr.gen_recordings(
                templates=self.templates_path,
                params=self.recordings_params,
                verbose=True,
                n_jobs=-1,
                drift_dicts=self._make_session_drift_dicts(i)
                )
            rec_path = self.output_dir / self.curr_run / f'recording{i+1}.h5'
            mr.save_recording_generator(recgen, filename=rec_path)
            logging.info(f"Generated and saved {self.n_recordings} recordings.")
            self.recgen = recgen

    def _make_session_drift_dicts(self, i):
        inter_session_drift_dict = self.config["base_signal_drift_dict"].copy()
        duration = self.recordings_params["duration"]
        drift_times = np.arange(0, duration, 0.1)
        depth = self.z_offset * i
        drift_vector = np.array([0, 0, depth])
        inter_session_drift_dict["drift_times"] = drift_times
        inter_session_drift_dict["drift_vector"] = drift_vector

        intra_session_drift_dict = self.config["slow_rigid_drift_dict"].copy()
        intra_session_drift_dict["drift_times"] = drift_times
        return [inter_session_drift_dict, intra_session_drift_dict]


    def _load_recordings(self):
        recgen = mr.load_recordings(self.rec_path)
        logging.info("Loaded pre-saved recordings.")
        self.recgen = recgen

def main():
    config_dict = load_config()
    setup_logging(config_dict)
    logging.info("===== Starting simulation. =====")
    simulator = Simulator(config=config_dict)
    simulator.run_simulation()
    logging.info("===== Simulation finished. =====")

if __name__ == "__main__":
    main()