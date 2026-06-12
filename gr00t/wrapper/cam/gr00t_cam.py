from gr00t.policy.gr00t_policy import Gr00tPolicy


class Gr00tCamPolicy(Gr00tPolicy):

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Transform Gr00t sim observation format and compute actions.

        This method transforms the flat observation format from Gr00t sim environments
        into the nested format expected by Gr00tPolicy, computes actions, and transforms
        them back to the flat format expected by Gr00t sim environments.

        Input format (Gr00t sim):
            - Flat keys: 'video.camera_name', 'state.state_name'
            - Language: tuple[str] (B,)

        Output format (Gr00t sim):
            - Flat keys: 'action.action_name'

        Args:
            observation: Flat observation dictionary from Gr00t sim environment
            options: Optional parameters (currently unused)

        Returns:
            Tuple of (flat_actions_dict, info_dict)
        """
        # Transform flat observation format to nested format expected by Gr00tPolicy
        breakpoint()
        new_obs = {}
        for modality in ["video", "state", "language"]:
            new_obs[modality] = {}
            for key in self.policy.modality_configs[modality].modality_keys:
                if modality == "language":
                    # PATCH: Legacy compatibility for DC environments
                    if key == "task" and "annotation.human.coarse_action" in observation:
                        parsed_key = "annotation.human.coarse_action"
                    # /PATCH
                    else:
                        parsed_key = key
                else:
                    # Construct flat key (e.g., 'video.camera' or 'state.joints')
                    parsed_key = f"{modality}.{key}"

                arr = observation[parsed_key]

                # Transform to nested format
                if modality == "language":
                    # Convert from tuple[str] or list[str] (B,) to list[list[str]] (B, 1)
                    # Each element becomes a list with one string for temporal dimension
                    new_obs[modality][key] = [[str(item)] for item in arr]
                else:
                    # Video and state arrays are already in correct format (B, T, ...)
                    new_obs[modality][key] = arr

        # Compute actions using the underlying Gr00tPolicy
        action, info = self.policy.get_action(new_obs, options)

        # Transform actions back to flat format for Gr00t sim environment
        # action['joints'] -> 'action.joints'
        return {f"action.{key}": action[key] for key in action}, info