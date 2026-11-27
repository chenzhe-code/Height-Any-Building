import re

PATCH_POSITION = re.compile(r'_x(\d+)_y(\d+)')


def patch_position_bounds(filenames=()):
    """Return the minimum and maximum pixel origins encoded in filenames."""
    max_x = float('-inf')
    max_y = float('-inf')
    min_x = float('inf')
    min_y = float('inf')

    for filename in filenames:
        position = parse_patch_position(filename)
        if position is not None:
            x, y = position
            min_x, max_x = min(min_x, x), max(max_x, x)
            min_y, max_y = min(min_y, y), max(max_y, y)

    return {'max_x': max_x, 'min_x': min_x, "max_y": max_y, "min_y": min_y}


def mosaic_origin(files_meta=()):
    min_x = float('inf')
    max_y = float('-inf')
    for meta in files_meta:
        transform = meta['transform']
        upper_left_x = transform.c
        upper_left_y = transform.f

        if upper_left_x < min_x:
            min_x = upper_left_x
        if upper_left_y > max_y:
            max_y = upper_left_y

    return {'origin_x': min_x, 'origin_y': max_y}


def parse_patch_position(filename):
    """Parse the pixel origin from a window filename."""
    match = PATCH_POSITION.search(filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None
