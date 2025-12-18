# This is a temporary file to append override_paths function
def override_paths(poly_segments=None, dot_segments=None):
    """
    Override global path data with external sources (e.g. from HalconProductManager).
    This allows multicolor PathSegment objects to replace legacy numpy arrays.
    
    Args:
        poly_segments: List of PathSegment objects or numpy arrays for polylines
        dot_segments: List of PathSegment objects or numpy arrays for dots
    """
    global _PATH_POLY, _PATH_DOTS
    
    if poly_segments is not None:
        _PATH_POLY = poly_segments
        print(f"[mold_detector] Path override: {len(poly_segments)} polyline segments")
    
    if dot_segments is not None:
        _PATH_DOTS = dot_segments
        print(f"[mold_detector] Path override: {len(dot_segments)} dot segments")
