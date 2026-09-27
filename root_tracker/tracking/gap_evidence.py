"""Physical-component evidence for newly rooted, disconnected objects."""
import numpy as np
from scipy.spatial import cKDTree


class GapEvidence:
    """Bound unsupported new links by the observed object's spatial extent.

    Graph endpoints can be far apart on one connected root. Measure separation
    between entire physical components instead, preserving those connections.
    This gate is for new assignments only; established ownership is independent
    evidence and is handled by the temporal router.
    """
    def __init__(self, uppers):
        grouped = {}
        self.components = {}
        for upper in uppers:
            component = upper.get('component_id')
            if component is None:
                continue
            grouped.setdefault(component, []).extend(upper['contour'][:, 0])
            self.components[upper['point']] = component
            self.components[upper['lower_point']] = component
        self.points = {key: np.asarray(points) for key, points in grouped.items()}
        self.trees = {key: cKDTree(points) for key, points in self.points.items()}
        self.extents = {key: float(np.linalg.norm(np.ptp(points, axis=0)))
                        for key, points in self.points.items()}
        for upper in uppers:
            component = upper.get('component_id')
            if component in self.extents:
                self.extents[component] = max(self.extents[component],
                                             upper.get('component_extent', 0.))
        self.elongated = set()
        self.measured = set()
        for upper in uppers:
            component = upper.get('component_id')
            area = upper.get('component_area')
            if component in self.extents and area:
                self.measured.add(component)
                # Compare extent with the diameter of an equal-area disk.
                # A span beyond two such diameters is tubular evidence even
                # when segmentation leaves a long gap. This dimensionless shape
                # check is unchanged by image scaling.
                if self.extents[component] > 2 * np.sqrt(4 * area / np.pi):
                    self.elongated.add(component)
        self.cache = {}

    def allows(self, upper, parent):
        component = upper.get('component_id')
        if component not in self.measured or component in self.elongated:
            return True
        parent_component = self.components.get(parent)
        if parent_component == component:
            return True
        key = (component, parent_component if parent_component is not None else parent)
        if key not in self.cache:
            if parent_component is not None:
                distance = self.trees[parent_component].query(self.points[component])[0].min()
            else:
                distance = self.trees[component].query(parent)[0]
            self.cache[key] = distance <= self.extents[component]
        return self.cache[key]
