import random
import pandas as pd

def generate_sample():
    room_w = round(random.uniform(5.0, 10.0), 1)
    room_h = round(random.uniform(5.0, 10.0), 1)
    
    sofa_w = round(random.uniform(1.0, 3.0), 1)
    sofa_h = round(random.uniform(1.0, 3.0), 1)
    table_w = round(random.uniform(1.0, 2.0), 1)
    table_h = round(random.uniform(1.0, 2.0), 1)
    chair_w = round(random.uniform(0.5, 1.5), 1)
    chair_h = round(random.uniform(0.5, 1.5), 1)
    
    # Place sofa along a wall
    wall = random.choice(['left', 'right', 'top', 'bottom'])
    if wall == 'left':
        sofa_x, sofa_y = 0.0, random.uniform(0, room_h - sofa_h)
    elif wall == 'right':
        sofa_x, sofa_y = room_w - sofa_w, random.uniform(0, room_h - sofa_h)
    elif wall == 'top':
        sofa_x, sofa_y = random.uniform(0, room_w - sofa_w), room_h - sofa_h
    else:
        sofa_x, sofa_y = random.uniform(0, room_w - sofa_w), 0.0
    
    # Place table centrally
    table_x = (room_w - table_w) / 2
    table_y = (room_h - table_h) / 2
    
    # Place chair near table
    direction = random.choice(['left', 'right', 'above', 'below'])
    spacing = 0.5
    if direction == 'left':
        chair_x = table_x - chair_w - spacing
        chair_y = table_y
    elif direction == 'right':
        chair_x = table_x + table_w + spacing
        chair_y = table_y
    elif direction == 'above':
        chair_x = table_x
        chair_y = table_y + table_h + spacing
    else:
        chair_x = table_x
        chair_y = table_y - chair_h - spacing
    
    # Ensure chair is within bounds
    chair_x = max(0.0, min(chair_x, room_w - chair_w))
    chair_y = max(0.0, min(chair_y, room_h - chair_h))
    
    return [
        room_w, room_h,
        sofa_w, sofa_h,
        table_w, table_h,
        chair_w, chair_h,
        sofa_x, sofa_y,
        table_x, table_y,
        chair_x, chair_y
    ]

# Generate dataset
data = [generate_sample() for _ in range(1000)]
columns = [
    'room_w', 'room_h',
    'sofa_w', 'sofa_h',
    'table_w', 'table_h',
    'chair_w', 'chair_h',
    'sofa_x', 'sofa_y',
    'table_x', 'table_y',
    'chair_x', 'chair_y'
]
df = pd.DataFrame(data, columns=columns)
df.to_csv('data/synthetic_data.csv', index=False)