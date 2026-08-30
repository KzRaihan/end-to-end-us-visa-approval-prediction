# ==============================================================================
# us_visa/entity/estimator.py
# Target Variable Encoder — Maps categorical target to numerical labels
# ==============================================================================
class TargetValueMapping:
    """Defines the mapping for the US Visa target variable."""

    def __init__(self):
        # Define target labels used during model training
        self.mapping = {
            "Certified": 0,
            "Denied": 1
        }

    def as_dict(self) -> dict:
        """Return the forward target mapping."""

        return self.mapping

    def reverse_mapping(self) -> dict:
        """Return numeric label to original target mapping."""

        return {
            value: key
            for key, value in self.mapping.items()
        }