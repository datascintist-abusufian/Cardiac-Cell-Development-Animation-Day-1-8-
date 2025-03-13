import streamlit as st
import time
import pandas as pd
import numpy as np
import altair as alt
from PIL import Image, ImageDraw
import io
import base64

st.set_page_config(page_title="Cardiac Cell Development Animation", layout="wide")

# Title and description
st.title("Cardiac Cell Development Animation (Day 1-8) by Abu Sufian")
st.markdown("This visualization shows the morphological and functional changes in cardiac cells over an 8-day period.")

# Cell data for each day
cell_data = {
    1: {
        "title": "Day 1: Immature Stage",
        "shape": "Small, round, loosely attached cells",
        "density": "Sparse distribution, minimal cell-cell interaction",
        "nucleus": "Large, prominent, occupying most of the cytoplasm",
        "cytoplasm": "Low actin filament density, no organized sarcomeres",
        "contractility": "Very weak or absent, minimal spontaneous twitching",
        "noise": "Low",
        "functional_state": "Highly immature, incapable of coordinated beating",
        "color": (255, 214, 204),  # Light pinkish
        "beat": 0.2,  # Very weak beat
        "sync_level": 0.1,  # Almost no synchronization
        "cell_count": 8,  # Fewer cells
        "debris_level": 0.1,  # Minimal debris
    },
    2: {
        "title": "Day 2: Initial Beating",
        "shape": "Slight elongation, cells begin forming small clusters",
        "density": "Moderate increase in cell-cell interaction",
        "nucleus": "Still prominent, but relative cytoplasmic volume increasing",
        "cytoplasm": "More structured, actin filaments start forming",
        "contractility": "Few healthy cells start mild beating, but not synchronized",
        "noise": "Low",
        "functional_state": "Early contractions observed, but weak and inconsistent",
        "color": (255, 204, 204),  # Light pink
        "beat": 0.4,  # Weak beat
        "sync_level": 0.2,  # Little synchronization
        "cell_count": 12,  # More cells
        "debris_level": 0.1,  # Minimal debris
    },
    3: {
        "title": "Day 3: Sparse Mean Beating Begins",
        "shape": "Cells elongate, slight alignment observed",
        "density": "Increased junction formation, more intercellular connectivity",
        "nucleus": "Starting to appear smaller relative to expanding cytoplasm",
        "cytoplasm": "Early sarcomere structures begin forming, weak striations visible",
        "contractility": "Few healthy cells begin to show mean beating, still uncoordinated",
        "noise": "Low",
        "functional_state": "Patchy contractions, but improved over Day 2",
        "color": (255, 204, 204),  # Light pink
        "beat": 0.5,  # Stronger beat
        "sync_level": 0.3,  # More synchronization
        "cell_count": 16,  # More cells forming
        "debris_level": 0.2,  # Slight increase in debris
    },
    4: {
        "title": "Day 4: Stronger Contractions in Some Cells",
        "shape": "More defined, elongated, and better aligned cells",
        "density": "High, beginning of monolayer-like structures",
        "nucleus": "Evenly distributed, organized within the cell",
        "cytoplasm": "Denser filaments, early Z-line structures",
        "contractility": "More healthy cells with mean beating, improved rhythmicity",
        "noise": "Low",
        "functional_state": "Early functional cardiomyocyte-like properties emerge",
        "color": (255, 153, 153),  # Medium pink
        "beat": 0.7,  # Medium-strong beat
        "sync_level": 0.5,  # Half synchronized
        "cell_count": 20,  # Higher density
        "debris_level": 0.2,  # Still low debris
    },
    5: {
        "title": "Day 5: Moderate Synchronization in Beating",
        "shape": "Well-elongated, aligned along parallel lines",
        "density": "High, forming strong intercellular junctions",
        "nucleus": "Less prominent, as cytoplasm grows in volume",
        "cytoplasm": "Well-formed sarcomeres with clear striations",
        "contractility": "Moderate contraction force, clear mean beating pattern",
        "noise": "Moderate",
        "functional_state": "Stronger contractions, beginning of synchronized function",
        "color": (255, 102, 102),  # Stronger pink
        "beat": 0.8,  # Strong beat
        "sync_level": 0.7,  # Good synchronization
        "cell_count": 24,  # High density
        "debris_level": 0.3,  # Moderate debris
    },
    6: {
        "title": "Day 6: Peak Contraction Activity",
        "shape": "Fully elongated, clear cardiomyocyte morphology",
        "density": "Strongly connected monolayer, peak cell-to-cell adhesion",
        "nucleus": "Evenly spread, well-integrated",
        "cytoplasm": "Densely packed sarcomeres, clear actin-myosin interactions",
        "contractility": "High contraction intensity, peak synchronization in mean beating",
        "noise": "Slightly increasing due to metabolic stress",
        "functional_state": "Highest functionality, optimal contraction rhythm",
        "color": (255, 51, 51),  # Bright red
        "beat": 1.0,  # Maximum beat
        "sync_level": 0.9,  # Highly synchronized
        "cell_count": 28,  # Maximum density
        "debris_level": 0.4,  # Increasing debris
    },
    7: {
        "title": "Day 7: Damage & Fragmentation Begins",
        "shape": "Fragmentation starts, some cells detach",
        "density": "Decreasing due to stress-induced detachment",
        "nucleus": "Some nuclei appear condensed or fragmented",
        "cytoplasm": "Signs of actin filament disassembly, disrupted sarcomeres",
        "contractility": "Weaker contractions, loss of synchronization, some dead zones",
        "noise": "High, increased debris from cell detachment",
        "functional_state": "Declining function, early damage evident",
        "color": (204, 51, 51),  # Darker red
        "beat": 0.6,  # Weakening beat
        "sync_level": 0.5,  # Losing synchronization
        "cell_count": 20,  # Decreasing density
        "debris_level": 0.7,  # High debris
    },
    8: {
        "title": "Day 8: Significant Cell Damage",
        "shape": "High fragmentation, cell integrity severely compromised",
        "density": "Significant cell loss, visible gaps in the network",
        "nucleus": "Some remain intact, others fragmented or missing",
        "cytoplasm": "Loss of sarcomere organization, widespread cellular breakdown",
        "contractility": "Very weak or absent, most cells cease contracting",
        "noise": "Extremely high, cell fragments and debris dominate the field",
        "functional_state": "Experiment ends as contraction ceases and cells deteriorate",
        "color": (153, 51, 51),  # Brownish red
        "beat": 0.2,  # Very weak beat
        "sync_level": 0.2,  # Almost no synchronization
        "cell_count": 12,  # Few remaining cells
        "debris_level": 0.9,  # Maximum debris
    }
}

# Create tabs for different views
tab1, tab2, tab3 = st.tabs(["Animation", "Cell Properties", "Data Visualization"])

with tab1:
    # Animation controls
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        auto_play = st.checkbox("Auto Play", value=False)
    
    with col2:
        day = st.slider("Select Day", min_value=1, max_value=8, value=1)
    
    with col3:
        speed = st.selectbox("Animation Speed", [1, 2, 3], index=1)
        frame_delay = 1.0 / speed
    
    # Session state to track animation
    if 'play_animation' not in st.session_state:
        st.session_state.play_animation = auto_play
        st.session_state.last_day = day
    
    # Update play state based on checkbox
    st.session_state.play_animation = auto_play
    
    if auto_play and st.session_state.last_day != day:
        # User manually changed the day slider, stop auto play
        st.session_state.play_animation = False
        auto_play = False
        st.session_state.last_day = day
    
    # Function to generate cell animation frame with more realistic morphology
    def generate_cell_frame(day_num, pulse=0.0):
        day_data = cell_data[day_num]
        width, height = 800, 600
        
        # Create a blank white image
        image = Image.new('RGB', (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        
        # Create cell clusters - cells tend to grow in groups
        num_clusters = max(3, day_num)
        cluster_centers = []
        for _ in range(num_clusters):
            cluster_centers.append((50 + np.random.random() * 700, 50 + np.random.random() * 500))
        
        # Draw connecting fibers between clusters (days 4-6)
        if 4 <= day_num <= 6:
            for i in range(len(cluster_centers)):
                for j in range(i+1, len(cluster_centers)):
                    # Only connect some clusters
                    if np.random.random() < 0.6:
                        x1, y1 = cluster_centers[i]
                        x2, y2 = cluster_centers[j]
                        # Draw thin connecting fibers between clusters
                        fiber_color = (255, 180, 180, 100)  # Light red, transparent
                        # Create a wavy line
                        points = []
                        steps = 10
                        for step in range(steps + 1):
                            t = step / steps
                            x = x1 + (x2 - x1) * t
                            y = y1 + (y2 - y1) * t
                            # Add wave effect
                            if 0 < step < steps:
                                wave_amp = 10 + 5 * np.sin(step)
                                x += np.sin(step * 3) * wave_amp
                                y += np.cos(step * 2) * wave_amp
                            points.append((x, y))
                        
                        # Draw the fiber line
                        for k in range(len(points) - 1):
                            draw.line([points[k], points[k+1]], fill=fiber_color, width=2)
        
        # Draw cells
        cells_drawn = 0
        clusters_used = 0
        
        while cells_drawn < day_data["cell_count"]:
            # Select a cluster to add cells to
            if clusters_used < len(cluster_centers):
                cluster_x, cluster_y = cluster_centers[clusters_used]
                clusters_used += 1
            else:
                # If we've used all clusters, pick a random one
                cluster_index = np.random.randint(0, len(cluster_centers))
                cluster_x, cluster_y = cluster_centers[cluster_index]
            
            # Determine how many cells to add to this cluster
            cells_in_cluster = min(
                max(2, int(day_data["cell_count"] / num_clusters + np.random.randint(-2, 3))),
                day_data["cell_count"] - cells_drawn
            )
            
            for _ in range(cells_in_cluster):
                # Calculate position within the cluster
                cluster_radius = 30 + day_num * 5
                angle = np.random.random() * 2 * np.pi
                distance = np.random.random() * cluster_radius
                x = cluster_x + np.cos(angle) * distance
                y = cluster_y + np.sin(angle) * distance
                
                # DAY 1: Small, round, immature cells
                if day_num == 1:
                    cell_width = 18 + np.random.random() * 7
                    cell_height = cell_width
                    cell_color = (255, 214, 204, 180)  # Light pinkish
                    
                    # Almost no beating
                    beat_factor = 1.0 + pulse * 0.05
                    cell_width *= beat_factor
                    cell_height *= beat_factor
                    
                    # Draw basic round cell
                    draw.ellipse([x, y, x + cell_width, y + cell_height], fill=cell_color)
                    
                    # Large nucleus
                    nucleus_size = 0.7 + np.random.random() * 0.1
                    nucleus_color = (102, 102, 204, 200)  # Blue nucleus
                
                # DAY 2: Slightly elongated, few healthy cells
                elif day_num == 2:
                    # 70% round cells, 30% slightly elongated
                    if np.random.random() < 0.7:
                        cell_width = 20 + np.random.random() * 8
                        cell_height = cell_width
                    else:
                        cell_width = 15 + np.random.random() * 8
                        cell_height = cell_width * (1.2 + np.random.random() * 0.3)
                    
                    # Weak beating in some cells
                    cell_color = (255, 204, 204, 180)  # Light pink
                    if np.random.random() < 0.4:  # Only 40% of cells beat
                        beat_factor = 1.0 + pulse * day_data["beat"] * 0.15
                        cell_width *= beat_factor
                        cell_height *= beat_factor
                    
                    # Draw basic cell
                    draw.ellipse([x, y, x + cell_width, y + cell_height], fill=cell_color)
                    
                    # Large nucleus but slightly smaller than day 1
                    nucleus_size = 0.65 + np.random.random() * 0.1
                    nucleus_color = (102, 102, 204, 200)  # Blue nucleus
                
                # DAY 3: More elongated, healthy but some aging
                elif day_num == 3:
                    # 40% round, 60% elongated
                    if np.random.random() < 0.4:
                        cell_width = 20 + np.random.random() * 8
                        cell_height = cell_width
                    else:
                        cell_width = 15 + np.random.random() * 8
                        cell_height = cell_width * (1.5 + np.random.random() * 0.5)
                        
                        # Rotation angle (simplified by skewing dimensions)
                        if np.random.random() < 0.5:
                            cell_width, cell_height = cell_height, cell_width
                    
                    # Beating in more cells
                    cell_color = (255, 194, 194, 180)  # Pink
                    if np.random.random() < 0.6:  # 60% of cells beat
                        beat_factor = 1.0 + pulse * day_data["beat"] * 0.2
                        cell_width *= beat_factor
                        cell_height *= beat_factor
                    
                    # Draw cell
                    draw.ellipse([x, y, x + cell_width, y + cell_height], fill=cell_color)
                    
                    # Add some internal structure (sarcomeres forming)
                    if np.random.random() < 0.4:
                        for i in range(3):
                            line_y = y + cell_height * (0.3 + i * 0.2)
                            line_length = cell_width * 0.6
                            line_x = x + (cell_width - line_length) / 2
                            draw.line([(line_x, line_y), (line_x + line_length, line_y)], 
                                     fill=(255, 160, 160, 120), width=1)
                    
                    # Medium sized nucleus
                    nucleus_size = 0.5 + np.random.random() * 0.1
                    nucleus_color = (102, 102, 204, 200)  # Blue nucleus
                
                # DAY 4: Well-defined, elongated, aligned cells
                elif day_num == 4:
                    # 20% round, 80% elongated
                    if np.random.random() < 0.2:
                        cell_width = 20 + np.random.random() * 8
                        cell_height = cell_width
                    else:
                        cell_width = 15 + np.random.random() * 8
                        cell_height = cell_width * (1.8 + np.random.random() * 0.7)
                        
                        # Rotation angle (simplified by skewing dimensions)
                        if np.random.random() < 0.5:
                            cell_width, cell_height = cell_height, cell_width
                    
                    # Better synchronized beating
                    cell_color = (255, 153, 153, 180)  # Medium pink
                    if np.random.random() < 0.7:  # 70% of cells beat
                        beat_factor = 1.0 + pulse * day_data["beat"] * 0.25
                        cell_width *= beat_factor
                        cell_height *= beat_factor
                    
                    # Draw cell
                    draw.ellipse([x, y, x + cell_width, y + cell_height], fill=cell_color)
                    
                    # Add internal structure (sarcomeres more visible)
                    if np.random.random() < 0.8:
                        lines = int(3 + np.random.random() * 3)
                        for i in range(lines):
                            line_y = y + cell_height * (0.2 + i * 0.6/lines)
                            line_length = cell_width * 0.8
                            line_x = x + (cell_width - line_length) / 2
                            draw.line([(line_x, line_y), (line_x + line_length, line_y)], 
                                     fill=(255, 130, 130, 150), width=1)
                    
                    # Smaller nucleus
                    nucleus_size = 0.45 + np.random.random() * 0.1
                    nucleus_color = (102, 102, 204, 200)  # Blue nucleus
                
                # DAY 5-6: Peak maturity, strong organization and connection
                elif day_num <= 6:
                    # 10% round, 90% elongated
                    if np.random.random() < 0.1:
                        cell_width = 20 + np.random.random() * 8
                        cell_height = cell_width
                    else:
                        cell_width = 15 + np.random.random() * 10
                        cell_height = cell_width * (2.0 + np.random.random() * 1.0)
                        
                        # More consistent alignment
                        if np.random.random() < 0.7:
                            cell_width, cell_height = cell_height, cell_width
                    
                    # Strong synchronized beating
                    intensity = 0.8 if day_num == 5 else 1.0  # Day 6 is peak activity
                    cell_color = (255, 102 - (day_num-5)*40, 102 - (day_num-5)*40, 180)  # Stronger red for day 6
                    if np.random.random() < 0.9:  # 90% of cells beat
                        beat_factor = 1.0 + pulse * day_data["beat"] * 0.3 * intensity
                        cell_width *= beat_factor
                        cell_height *= beat_factor
                    
                    # Draw cell
                    draw.ellipse([x, y, x + cell_width, y + cell_height], fill=cell_color)
                    
                    # Add detailed internal structure (well-formed sarcomeres)
                    lines = int(5 + np.random.random() * 3)
                    for i in range(lines):
                        line_y = y + cell_height * (0.2 + i * 0.6/lines)
                        line_length = cell_width * 0.85
                        line_x = x + (cell_width - line_length) / 2
                        draw.line([(line_x, line_y), (line_x + line_length, line_y)], 
                                 fill=(255, 80, 80, 180), width=2)
                    
                    # Add intercellular connections
                    if np.random.random() < 0.4:
                        connection_x = x + cell_width
                        connection_y = y + cell_height/2
                        conn_length = 10 + np.random.random() * 15
                        draw.line([(connection_x, connection_y), (connection_x + conn_length, connection_y)], 
                                 fill=(255, 120, 120, 150), width=2)
                    
                    # Well-integrated nucleus
                    nucleus_size = 0.4
                    nucleus_color = (102, 102, 204, 200)  # Blue nucleus
                
                # DAY 7: Beginning of damage and fragmentation
                elif day_num == 7:
                    # Mix of healthy, fragmenting, and detaching cells
                    cell_state = np.random.random()
                    if cell_state < 0.4:  # 40% still relatively healthy
                        cell_width = 15 + np.random.random() * 10
                        cell_height = cell_width * (1.8 + np.random.random() * 0.5)
                        cell_color = (204, 51, 51, 160)  # Darker red, more transparent
                        
                        # Some beating, but weaker
                        if np.random.random() < 0.6:  # 60% of "healthy" cells still beat
                            beat_factor = 1.0 + pulse * day_data["beat"] * 0.15
                            cell_width *= beat_factor
                            cell_height *= beat_factor
                            
                        # Cell membrane starting to break down
                        if np.random.random() < 0.5:
                            # Add "breaks" in the membrane
                            break_angle = np.random.random() * 2 * np.pi
                            break_size = np.random.random() * 5 + 3
                            break_x = x + cell_width/2 + np.cos(break_angle) * cell_width/2
                            break_y = y + cell_height/2 + np.sin(break_angle) * cell_height/2
                            draw.ellipse([break_x-break_size/2, break_y-break_size/2, 
                                         break_x+break_size/2, break_y+break_size/2], 
                                        fill=(255, 255, 255, 255))  # White "break"
                    
                    elif cell_state < 0.7:  # 30% fragmenting
                        # Draw multiple smaller fragments instead of one cell
                        fragment_count = int(2 + np.random.random() * 3)
                        for j in range(fragment_count):
                            frag_x = x + np.random.random() * 20 - 10
                            frag_y = y + np.random.random() * 20 - 10
                            frag_size = 6 + np.random.random() * 8
                            frag_color = (204, 51, 51, 140 - j*20)  # Progressively more transparent
                            draw.ellipse([frag_x, frag_y, frag_x+frag_size, frag_y+frag_size], 
                                        fill=frag_color)
                        
                        # Skip standard cell drawing
                        cells_drawn += 1
                        continue
                    
                    else:  # 30% severely damaged/detaching
                        cell_width = 12 + np.random.random() * 8
                        cell_height = 12 + np.random.random() * 8
                        cell_color = (180, 40, 40, 120)  # Dark red, very transparent
                        
                        # No beating for severely damaged cells
                        beat_factor = 1.0
                        
                        # Cell border is irregular
                        draw.ellipse([x, y, x + cell_width, y + cell_height], fill=cell_color)
                        
                        # Add cellular debris around damaged cells
                        debris_count = int(3 + np.random.random() * 5)
                        for j in range(debris_count):
                            debris_x = x + np.random.random() * (cell_width + 20) - 10
                            debris_y = y + np.random.random() * (cell_height + 20) - 10
                            debris_size = 2 + np.random.random() * 3
                            draw.ellipse([debris_x, debris_y, debris_x+debris_size, debris_y+debris_size], 
                                        fill=(150, 50, 50, 100 + int(np.random.random() * 50)))
                        
                        # Skip nucleus for severely damaged cells
                        cells_drawn += 1
                        continue
                    
                    # Draw cell (for healthy and some fragmenting cells)
                    draw.ellipse([x, y, x + cell_width, y + cell_height], fill=cell_color)
                    
                    # Degraded internal structure
                    if np.random.random() < 0.4:
                        for i in range(2):
                            line_y = y + cell_height * (0.3 + i * 0.3)
                            line_length = cell_width * 0.5
                            line_x = x + (cell_width - line_length) / 2
                            # Broken lines
                            segments = 3
                            for s in range(segments):
                                if np.random.random() < 0.7:  # Some segments missing
                                    seg_start = line_x + (line_length * s / segments)
                                    seg_end = line_x + (line_length * (s+1) / segments)
                                    draw.line([(seg_start, line_y), (seg_end, line_y)], 
                                             fill=(200, 70, 70, 120), width=1)
                    
                    # Nucleus sometimes fragmented or condensed
                    if np.random.random() < 0.5:
                        nucleus_size = 0.3 + np.random.random() * 0.1
                        nucleus_color = (102, 102, 204, 120)  # More transparent
                    else:
                        # Fragmented nucleus - draw multiple small pieces
                        for j in range(2):
                            nuc_x = x + cell_width * (0.3 + np.random.random() * 0.4)
                            nuc_y = y + cell_height * (0.3 + np.random.random() * 0.4)
                            nuc_size = cell_width * 0.2
                            draw.ellipse([nuc_x, nuc_y, nuc_x+nuc_size, nuc_y+nuc_size], 
                                        fill=(102, 102, 204, 100))
                        
                        # Skip standard nucleus drawing
                        cells_drawn += 1
                        continue
                
                # DAY 8: Severe damage and cell death
                else:  # day_num == 8
                    # Mostly debris and fragments with very few intact cells
                    cell_state = np.random.random()
                    if cell_state < 0.2:  # Only 20% somewhat intact
                        cell_width = 10 + np.random.random() * 8
                        cell_height = cell_width * (1.0 + np.random.random() * 0.3)
                        cell_color = (153, 51, 51, 130)  # Brownish red, very transparent
                        
                        # Almost no beating
                        if np.random.random() < 0.2:  # 20% of remaining cells beat weakly
                            beat_factor = 1.0 + pulse * day_data["beat"] * 0.1
                            cell_width *= beat_factor
                            cell_height *= beat_factor
                        
                        # Draw damaged cell
                        draw.ellipse([x, y, x + cell_width, y + cell_height], fill=cell_color)
                        
                        # Severely disrupted structure - just random dots inside
                        dots = int(2 + np.random.random() * 3)
                        for i in range(dots):
                            dot_x = x + np.random.random() * cell_width
                            dot_y = y + np.random.random() * cell_height
                            dot_size = 1 + np.random.random() * 2
                            draw.ellipse([dot_x, dot_y, dot_x+dot_size, dot_y+dot_size], 
                                        fill=(180, 60, 60, 150))
                        
                        # Some nuclei still visible but condensed
                        nucleus_size = 0.25
                        nucleus_color = (102, 102, 204, 80)  # Very faint
                    
                    else:  # 80% fragmented/debris
                        # Draw multiple smaller fragments
                        fragment_count = int(1 + np.random.random() * 5)
                        for j in range(fragment_count):
                            frag_x = x + np.random.random() * 30 - 15
                            frag_y = y + np.random.random() * 30 - 15
                            frag_size = 3 + np.random.random() * 6
                            frag_color = (153, 51, 51, 100 - j*10)  # Progressively more transparent
                            draw.ellipse([frag_x, frag_y, frag_x+frag_size, frag_y+frag_size], 
                                        fill=frag_color)
                        
                        # Skip standard cell and nucleus drawing
                        cells_drawn += 1
                        continue
                
                # Draw nucleus unless skipped in special cases above
                nucleus_x = x + (cell_width / 2) - (cell_width * nucleus_size / 2)
                nucleus_y = y + (cell_height / 2) - (cell_height * nucleus_size / 2)
                nucleus_width = cell_width * nucleus_size
                nucleus_height = cell_height * nucleus_size
                draw.ellipse([nucleus_x, nucleus_y, nucleus_x + nucleus_width, nucleus_y + nucleus_height], 
                             fill=nucleus_color)
                
                cells_drawn += 1
        
        # Add additional debris and cellular fragments
        # More debris in later days
        base_debris = int(day_data["debris_level"] * 100)
        for i in range(base_debris):
            debris_x = np.random.random() * width
            debris_y = np.random.random() * height
            debris_size = 2 + np.random.random() * 4
            
            # Debris color varies by day
            if day_num <= 3:
                debris_color = (180, 180, 180, 80)  # Light gray, very transparent
            elif day_num <= 6:
                debris_color = (180, 150, 150, 100)  # Pinkish gray
            else:
                debris_color = (160, 100, 100, 120)  # Reddish debris for cell breakdown
            
            draw.ellipse([debris_x, debris_y, debris_x + debris_size, debris_y + debris_size], 
                         fill=debris_color)
        
        return image
    
    # Display the current day's data
    current_day_data = cell_data[day]
    st.subheader(current_day_data["title"])

    # Create animation placeholder
    animation_placeholder = st.empty()
    
    # Manually advance day for auto-play
    if auto_play:
        # Create 10 frames with pulsing effect for the current day
        for pulse in np.linspace(0, 1, 10):
            if not st.session_state.play_animation:
                break
                
            frame = generate_cell_frame(day, pulse)
            
            # Convert PIL image to bytes for display
            buf = io.BytesIO()
            frame.save(buf, format="PNG")
            byte_im = buf.getvalue()
            
            # Display using Streamlit image
            animation_placeholder.image(byte_im, use_container_width=True)
            
            # Control frame rate
            time.sleep(frame_delay/10)
            
        # Move to next day if auto-playing
        if st.session_state.play_animation:
            next_day = day + 1 if day < 8 else 1
            st.session_state.last_day = next_day
            time.sleep(frame_delay)
            # Replace st.experimental_rerun() with st.rerun()
            st.rerun()
    else:
        # Just show a static frame with a slight pulse
        frame = generate_cell_frame(day, 0.5)
        
        # Convert PIL image to bytes for display
        buf = io.BytesIO()
        frame.save(buf, format="PNG")
        byte_im = buf.getvalue()
        
        # Display using Streamlit image
        animation_placeholder.image(byte_im, use_container_width=True)

with tab2:
    # Display cell properties
    st.subheader(f"Cell Properties - {cell_data[day]['title']}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Morphology")
        st.markdown(f"**Shape:** {cell_data[day]['shape']}")
        st.markdown(f"**Density:** {cell_data[day]['density']}")
        st.markdown(f"**Nucleus:** {cell_data[day]['nucleus']}")
        st.markdown(f"**Cytoplasm:** {cell_data[day]['cytoplasm']}")
    
    with col2:
        st.markdown("### Functionality")
        st.markdown(f"**Contractility:** {cell_data[day]['contractility']}")
        st.markdown(f"**Noise/Debris:** {cell_data[day]['noise']}")
        st.markdown(f"**Functional State:** {cell_data[day]['functional_state']}")

with tab3:
    # Data visualization
    st.subheader("Quantitative Changes Over Time")
    
    # Create a dataframe with the numeric data
    metrics = ["beat", "sync_level", "cell_count", "debris_level"]
    df_data = {
        "Day": list(range(1, 9)),
        "Beat Strength": [cell_data[d]["beat"] * 100 for d in range(1, 9)],
        "Synchronization": [cell_data[d]["sync_level"] * 100 for d in range(1, 9)],
        "Cell Count": [cell_data[d]["cell_count"] for d in range(1, 9)],
        "Debris Level": [cell_data[d]["debris_level"] * 100 for d in range(1, 9)]
    }
    df = pd.DataFrame(df_data)
    
    # Create metric selection
    selected_metrics = st.multiselect(
        "Select metrics to display",
        ["Beat Strength", "Synchronization", "Cell Count", "Debris Level"],
        default=["Beat Strength", "Synchronization", "Cell Count"]
    )
    
    if selected_metrics:
        # Create a long-form dataframe for Altair
        df_long = pd.melt(
            df, 
            id_vars=["Day"], 
            value_vars=selected_metrics,
            var_name="Metric", 
            value_name="Value"
        )
        
        # Create the chart
        chart = alt.Chart(df_long).mark_line(point=True).encode(
            x=alt.X("Day:O", title="Day"),
            y=alt.Y("Value:Q", title="Value"),
            color=alt.Color("Metric:N", title="Metric"),
            tooltip=["Day", "Metric", "Value"]
        ).properties(
            width=700,
            height=400,
            title="Cell Development Metrics Over Time"
        ).interactive()
        
        st.altair_chart(chart, use_container_width=True)
        
        # Show the data table
        st.dataframe(df[["Day"] + selected_metrics])
    else:
        st.info("Please select at least one metric to display")

# Add app instructions
st.sidebar.header("Instructions")
st.sidebar.markdown("""
This app visualizes the development of cardiac cells from Day 1 to Day 8.

### How to use:
1. Use the slider to select the day
2. Toggle auto-play to run the animation
3. Adjust animation speed as needed
4. Switch between tabs to see different views:
   - Animation: Visual representation of cells
   - Cell Properties: Detailed information about each day
   - Data Visualization: Quantitative metrics over time

### About the data:
This visualization is based on a study of cardiac cell development, showing morphological and functional changes over an 8-day period, from immature cells (Day 1) to peak activity (Day 6) and subsequent deterioration (Days 7-8).
""")

# Add data source information 
st.sidebar.markdown("---")
st.sidebar.markdown("Data source: Cardiac Cell Development Study")
