from pyvene import (
    IntervenableConfig,
    RepresentationConfig,
    BoundlessRotatedSpaceIntervention,
    LowRankRotatedSpaceIntervention,
    RotatedSpaceIntervention,
    CollectIntervention,
)
from interventions.interventions import LowRankInvertibleRealNVPIntervention, StretchedSpaceIntervention, RealNVPIntervention, \
    RevNetIntervention, UnconstrainedIntervention, LowRankInvertibleUnconstrainedIntervention, LowRankInvertibleRevNetIntervention,\
    LowRankInvertibleRotatedIntervention, LowRankRotatedSpaceInterventionCustomDim, LowRankInvertibleUnconstrained2DIntervention

def low_rank_custom_dim_config(model_type, intervention_type, layer, low_rank_dimension=1, intervention_dim=1):
    """
    Creates a low rank DAS configuration with custom intervention dimension.
    """
    def custom_dim_factory(**kwargs):
        kwargs.pop('intervention_dim', None)
        return LowRankRotatedSpaceInterventionCustomDim(intervention_dim=intervention_dim, **kwargs)

    config = IntervenableConfig(
        model_type=model_type,
        representations=[
            RepresentationConfig(
                layer=layer,
                component=intervention_type,
                low_rank_dimension=low_rank_dimension,
            ),
        ],
        intervention_types=custom_dim_factory,
    )
    return config, "rotate_layer"

def simple_low_rank_das_position_config(model_type, intervention_type, layer, low_rank_dimension=1):
    """
    Creates a low rank DAS configuration matching the notebook setup.
    """
    config = IntervenableConfig(
        model_type=model_type,
        representations=[
            RepresentationConfig(
                layer=layer,              # layer
                component=intervention_type,   # intervention type
                low_rank_dimension=low_rank_dimension,
            ),
        ],
        intervention_types=LowRankRotatedSpaceIntervention,
    )
    return config, "rotate_layer"

def low_rank_invertible_unconstrained_config(model_type, intervention_type, layer, low_rank_dimension=1):
    """
    Creates a low rank invertible unconstrained configuration.
    """
    config = IntervenableConfig(
        model_type=model_type,
        representations=[
            RepresentationConfig(
                layer=layer,              # layer
                component=intervention_type,   # intervention type
                low_rank_dimension=2, # hardcode to 2 for visualization purposes
            ),
        ],
        intervention_types=LowRankInvertibleUnconstrainedIntervention,
    )
    return config, "rotate_layer"


def low_rank_invertible_unconstrained_2d_config(model_type, intervention_type, layer, low_rank_dimension=1):
    """
    Creates a low rank invertible unconstrained configuration with 2D intervention.
    """
    config = IntervenableConfig(
        model_type=model_type,
        representations=[
            RepresentationConfig(
                layer=layer,              # layer
                component=intervention_type,   # intervention type
                low_rank_dimension=2, # hardcode to 2 for visualization purposes
            ),
        ],
        intervention_types=LowRankInvertibleUnconstrained2DIntervention,
    )
    return config, "rotate_layer"


def low_rank_invertible_realnvp_config(model_type, intervention_type, layer, low_rank_dimension=1, n_layers=None, latent_dim=None, hidden_dim=None):
    """
    Creates a low rank invertible RealNVP configuration with a factory to bind
    RealNVP hyperparameters.
    """
    def realnvp_factory(**kwargs):
        # Remove possible duplicates from kwargs and bind our hyperparameters.
        kwargs.pop('n_layers', None)
        kwargs.pop('latent_dim', None)
        kwargs.pop('hidden_dim', None)
        return LowRankInvertibleRealNVPIntervention(n_layers=n_layers, latent_dim=latent_dim, hidden_dim=hidden_dim, **kwargs)

    config = IntervenableConfig(
        model_type=model_type,
        representations=[
            RepresentationConfig(
                layer=layer,
                component=intervention_type,
                low_rank_dimension=2,  # allow caller to control
            ),
        ],
        intervention_types=realnvp_factory,
    )
    return config, "rotate_layer"


def low_rank_invertible_revnet_config(model_type, intervention_type, layer, low_rank_dimension=1, n_layers=None, latent_dim=None, hidden_dim=None):
    """
    Creates a low rank invertible RevNet configuration with a factory to bind
    RevNet hyperparameters.
    """
    def revnet_factory(**kwargs):
        kwargs.pop('n_layers', None)
        kwargs.pop('latent_dim', None)
        kwargs.pop('hidden_dim', None)
        return LowRankInvertibleRevNetIntervention(n_layers=n_layers, latent_dim=latent_dim, hidden_dim=hidden_dim, **kwargs)

    config = IntervenableConfig(
        model_type=model_type,
        representations=[
            RepresentationConfig(
                layer=layer,
                component=intervention_type,
                low_rank_dimension=2,  # allow caller to control
            ),
        ],
        intervention_types=revnet_factory,
    )
    return config, "rotate_layer"


def low_rank_invertible_rotated_config(model_type, intervention_type, layer, low_rank_dimension=1):
    """
    Creates a low rank invertible rotated configuration.
    """
    config = IntervenableConfig(
        model_type=model_type,
        representations=[
            RepresentationConfig(
                layer=layer,              # layer
                component=intervention_type,   # intervention type
                low_rank_dimension=2, # hardcode to 2 for visualization purposes
            ),
        ],
        intervention_types=LowRankInvertibleRotatedIntervention,
    )
    return config, "rotate_layer"


def simple_boundless_das_position_config(model_type, intervention_type, layer):
    """
    Creates a boundless DAS configuration matching the notebook setup.
    """
    config = IntervenableConfig(
        model_type=model_type,
        representations=[
            RepresentationConfig(
                layer=layer,              # layer
                component=intervention_type,  # intervention type
            ),
        ],
        intervention_types=BoundlessRotatedSpaceIntervention,
    )
    return config, "rotate_layer"


def simple_das_position_config(model_type, intervention_type, layer):
    """
    Creates a standard DAS configuration matching the notebook setup.
    """
    config = IntervenableConfig(
        model_type=model_type,
        representations=[
            RepresentationConfig(
                layer=layer,              # layer
                component=intervention_type,  # intervention type
            ),
        ],
        intervention_types=RotatedSpaceIntervention,
    )
    return config, "rotate_layer"


def unconstrained_config(model, layer, intervention_type):
    config = IntervenableConfig(
        model_type=type(model),
        representations=[
            RepresentationConfig(
                layer=layer,
                component=intervention_type
            ),
        ],
        intervention_types=UnconstrainedIntervention,
    )
    return config, "unconstrained_layer"


def stretch_config(model, layer, intervention_type):
    config = IntervenableConfig(
        model_type=type(model),
        representations=[
            RepresentationConfig(
                layer=layer,
                component=intervention_type
            ),
        ],
        intervention_types=StretchedSpaceIntervention,
    )
    return config, "stretch_layer"


def realnvp_config(model, layer, intervention_type, n_layers, latent_dim, hidden_dim=None):

    # Provide a factory callable that will be invoked by pyvene to create
    # an intervention instance. Passing an instance here would make pyvene
    # try to call the module instance (which forwards to its `forward`),
    # producing a TypeError. A factory avoids that by returning an object.

    def realnvp_factory(**kwargs):
        # Pop the specific hyperparameters to avoid passing them twice
        kwargs.pop('n_layers', None)
        kwargs.pop('latent_dim', None)
        kwargs.pop('hidden_dim', None)
        # bind our supplied hyperparameters but allow pyvene to pass
        # additional metadata via kwargs
        return RealNVPIntervention(n_layers=n_layers, latent_dim=latent_dim, hidden_dim=hidden_dim, **kwargs)

    config = IntervenableConfig(
        model_type=type(model),
        representations=[
            RepresentationConfig(
                layer=layer,
                component=intervention_type
            )
        ],
        intervention_types=realnvp_factory,
    )
    return config, "realNVPlayer"


def revnet_config(model, layer, intervention_type, n_layers, latent_dim, hidden_dim=None):
    # Use a factory callable so pyvene will call this to construct the
    # intervention instance with the proper hyperparameters.
    def revnet_factory(**kwargs):
        kwargs.pop('n_layers', None)
        kwargs.pop('latent_dim', None)
        kwargs.pop('hidden_dim', None)
        return RevNetIntervention(n_layers=n_layers, latent_dim=latent_dim, hidden_dim=hidden_dim, **kwargs)

    config = IntervenableConfig(
        model_type=type(model),
        representations=[
            RepresentationConfig(
                layer=layer,
                component=intervention_type
            )
        ],
        intervention_types=revnet_factory,
    )
    return config, "revNetLayer"


def get_config(intervention_name, model_type, layer, component="block_output", low_rank_dimension=1, **kwargs):
    """
    Unified configuration creation matching the notebook's approach.

    Args:
        intervention_name: One of 'low_rank', 'boundless', 'das'
        model_type: The model type (usually the class from the model object)
        layer: Layer number for the intervention
        component: Component type (default: "block_output")
        low_rank_dimension: Dimension for low rank interventions (default: 1)
    """
    if intervention_name == 'low_rank':
        return simple_low_rank_das_position_config(
            model_type=model_type,
            intervention_type=component,
            layer=layer,
            low_rank_dimension=low_rank_dimension
        )
    elif intervention_name == 'low_rank_invertible_unconstrained':
        return low_rank_invertible_unconstrained_config(
            model_type=model_type,
            intervention_type=component,
            layer=layer,
            low_rank_dimension=low_rank_dimension
        )
    elif intervention_name == 'low_rank_invertible_unconstrained_2d':
        return low_rank_invertible_unconstrained_2d_config(
            model_type=model_type,
            intervention_type=component,
            layer=layer,
            low_rank_dimension=low_rank_dimension
        )
    elif intervention_name == 'boundless':
        return simple_boundless_das_position_config(
            model_type=model_type,
            intervention_type=component,
            layer=layer
        )
    elif intervention_name == 'das':
        return simple_das_position_config(
            model_type=model_type,
            intervention_type=component,
            layer=layer
        )
    elif intervention_name == 'unconstrained':
        return unconstrained_config(
            model=model_type,
            layer=layer,
            intervention_type=component
        )
    elif intervention_name == 'stretch':
        return stretch_config(
            model=model_type,
            layer=layer,
            intervention_type=component
        )
    elif intervention_name == 'realnvp':
        assert 'n_layers' in kwargs and 'latent_dim' in kwargs and 'hidden_dim' in kwargs,\
              "n_layers, latent_dim, and hidden_dim must be provided for RealNVP configuration."
        return realnvp_config(
            model=model_type,
            layer=layer,
            intervention_type=component,
            n_layers=kwargs['n_layers'],
            latent_dim=kwargs['latent_dim'],
            hidden_dim=kwargs['hidden_dim']
        )
    elif intervention_name == 'revnet':
        assert 'n_layers' in kwargs and 'latent_dim' in kwargs and 'hidden_dim' in kwargs,\
              "n_layers, latent_dim, and hidden_dim must be provided for RevNet configuration."
        return revnet_config(
            model=model_type,
            layer=layer,
            intervention_type=component,
            n_layers=kwargs['n_layers'],
            latent_dim=kwargs['latent_dim'],
            hidden_dim=kwargs['hidden_dim']
        )
    elif intervention_name == 'low_rank_invertible_realnvp':
        assert 'n_layers' in kwargs and 'latent_dim' in kwargs and 'hidden_dim' in kwargs,\
              "n_layers, latent_dim, and hidden_dim must be provided for low-rank invertible RealNVP configuration."
        return low_rank_invertible_realnvp_config(
            model_type=model_type,
            intervention_type=component,
            layer=layer,
            low_rank_dimension=low_rank_dimension,
            n_layers=kwargs['n_layers'],
            latent_dim=kwargs['latent_dim'],
            hidden_dim=kwargs['hidden_dim']
        )
    elif intervention_name == 'low_rank_invertible_revnet':
        assert 'n_layers' in kwargs and 'latent_dim' in kwargs and 'hidden_dim' in kwargs,\
              "n_layers, latent_dim, and hidden_dim must be provided for low-rank invertible RevNet configuration."
        return low_rank_invertible_revnet_config(
            model_type=model_type,
            intervention_type=component,
            layer=layer,
            low_rank_dimension=low_rank_dimension,
            n_layers=kwargs['n_layers'],
            latent_dim=kwargs['latent_dim'],
            hidden_dim=kwargs['hidden_dim']
        )
    elif intervention_name == 'low_rank_invertible_rotated':
        return low_rank_invertible_rotated_config(
            model_type=model_type,
            intervention_type=component,
            layer=layer,
            low_rank_dimension=low_rank_dimension
        )
    elif intervention_name == 'low_rank_custom_dim':
        intervention_dim = kwargs.get('intervention_dim', 1)
        return low_rank_custom_dim_config(
            model_type=model_type,
            intervention_type=component,
            layer=layer,
            low_rank_dimension=low_rank_dimension,
            intervention_dim=intervention_dim
        )
    else:
        raise ValueError(f"Unsupported intervention type: {intervention_name}")

# Legacy function for backward compatibility
def das_config(model_type, target_representation, intervention_types):
    config = IntervenableConfig(
        model_type=model_type,
        representations=[
            target_representation
        ],
        intervention_types=intervention_types,
    )
    return config, "rotate_layer"


def collect_config(model_type, component, layer):
    """Create a simple position-based collection configuration."""
    config = IntervenableConfig(
        model_type=model_type,
        representations=[
            RepresentationConfig(
                layer,              # layer
                component,          # component
                "pos",              # intervention unit
            ),
        ],
        intervention_types=CollectIntervention
    )
    return config
