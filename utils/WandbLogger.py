class WandbLogger:
    def __init__(self, mission_type, destruction_factor=None):
        self.mission_type = mission_type
        self.destruction_factor = destruction_factor
        self.mission_log_name = "mission_type: " + str(mission_type) + " "
        if destruction_factor is not None:
            self.full_log_name = "d_factor: " + str(destruction_factor) + " " + str(mission_type) + " "
        else:
            self.full_log_name = self.mission_log_name
        self.logging_values = {}


    def add_logging_value(self, value_key, value):
        self.logging_values[str(self.full_log_name) + str(value_key)] = value
        self.logging_values[str(self.mission_log_name) + str(value_key)] = value
        if self.mission_type != "zero-shot":
            self.logging_values["woodPLUSdirt: " + str(value_key)] = value

    def add_logging_value_single(self, value_key, value):
        self.logging_values[str(self.full_log_name) + str(value_key)] = value

    def get_logging_values(self):
        return self.logging_values
