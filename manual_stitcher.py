# manual_stitcher.py
MANUAL_SNAP_OVERLAP = 1
def perform_manual_snap(piece1, edge1, piece2, edge2):
    dx, dy = 0, 0
    if edge1 == 'right' and edge2 == 'left':
        target_x = piece1.x + piece1.w - MANUAL_SNAP_OVERLAP
        target_y = piece1.y + (piece1.h - piece2.h) / 2
        dx, dy = target_x - piece2.x, target_y - piece2.y
    elif edge1 == 'left' and edge2 == 'right':
        target_x = piece1.x - piece2.w + MANUAL_SNAP_OVERLAP
        target_y = piece1.y + (piece1.h - piece2.h) / 2
        dx, dy = target_x - piece2.x, target_y - piece2.y
    elif edge1 == 'bottom' and edge2 == 'top':
        target_x = piece1.x + (piece1.w - piece2.w) / 2
        target_y = piece1.y + piece1.h - MANUAL_SNAP_OVERLAP
        dx, dy = target_x - piece2.x, target_y - piece2.y
    elif edge1 == 'top' and edge2 == 'bottom':
        target_x = piece1.x + (piece1.w - piece2.w) / 2
        target_y = piece1.y - piece2.h + MANUAL_SNAP_OVERLAP
        dx, dy = target_x - piece2.x, target_y - piece2.y
    else:
        print(f"Invalid snap combination: cannot snap '{edge1}' of piece1 to '{edge2}' of piece2.")
        return None
    return (dx, dy)