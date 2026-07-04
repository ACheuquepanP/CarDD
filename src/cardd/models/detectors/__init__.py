"""CarDD package."""
from cardd.models.detectors import mask_rcnn            # noqa: F401  base primero (registra CarddMaskRCNN)
from cardd.models.detectors import mask_rcnn_focal      # noqa: F401
from cardd.models.detectors import mask_rcnn_pointrend  # noqa: F401
from cardd.models.detectors import cascade_mask_rcnn    # noqa: F401
from cardd.models.detectors import mask_rcnn_cldice  # noqa: F401


# en src/cardd/models/detectors/__init__.py
from cardd.models.detectors import mask_rcnn_skeleton  # noqa: F401
from cardd.models.detectors import mask_rcnn_composed  # noqa: F401
from cardd.models.detectors import cascade_mask_rcnn_composed  # noqa: F401