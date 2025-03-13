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
    
    # Function to generate cell animation frame
    def generate_cell_frame(day_num, pulse=0.0):
        day_data = cell_data[day_num]
        width, height = 800, 600
        
        # Create a blank white image
        image = Image.new('RGB', (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        
        # Draw cells
        for i in range(day_data["cell_count"]):
            # Calculate position with some randomness
            x = 50 + np.random.random() * 700
            y = 50 + np.random.random() * 500
            
            # Calculate cell shape - early days are round, middle days elongated, late days fragmented
            if day_num <= 2:
                # Round cells
                cell_width = 20 + np.random.random() * 10
                cell_height = cell_width
                # Add beating effect
                if day_data["beat"] > 0.1:
                    beat_factor = 1.0 + pulse * day_data["beat"] * 0.2
                    cell_width *= beat_factor
                    cell_height *= beat_factor
            elif day_num <= 6:
                # Elongated cells
                cell_width = 15 + np.random.random() * 10
                cell_height = 30 + np.random.random() * 20 * (day_num/4)  # Gets more elongated
                # Add beating effect
                if day_data["beat"] > 0.1:
                    beat_factor = 1.0 + pulse * day_data["beat"] * 0.2
                    cell_width *= beat_factor
                    cell_height *= beat_factor
                # Rotate cells
                # Note: For simplicity, we're not rotating in this version
            else:
                # Fragmented cells
                cell_width = 10 + np.random.random() * 15
                cell_height = 20 + np.random.random() * 15
                # Less beating in damaged cells
                if day_data["beat"] > 0.1:
                    beat_factor = 1.0 + pulse * day_data["beat"] * 0.1
                    cell_width *= beat_factor
                    cell_height *= beat_factor
            
            # Draw cell with a slight transparency
            color_with_alpha = (*day_data["color"], 180)  # Add alpha for 70% opacity
            draw.ellipse([x, y, x + cell_width, y + cell_height], fill=color_with_alpha)
            
            # Draw nucleus
            nucleus_size = 0.7 if day_num <= 2 else 0.5 if day_num <= 6 else 0.3
            nucleus_x = x + (cell_width / 2) - (cell_width * nucleus_size / 2)
            nucleus_y = y + (cell_height / 2) - (cell_height * nucleus_size / 2)
            nucleus_width = cell_width * nucleus_size
            nucleus_height = cell_height * nucleus_size
            
            # Nucleus color - blue with transparency
            nucleus_color = (102, 102, 204, 200)  # Slightly transparent blue
            if day_num >= 7:  # Fading nucleus in later days
                nucleus_color = (102, 102, 204, 100)  # More transparent
                
            draw.ellipse([nucleus_x, nucleus_y, nucleus_x + nucleus_width, nucleus_y + nucleus_height], 
                          fill=nucleus_color)
        
        # Add debris based on debris level
        for i in range(int(day_data["debris_level"] * 100)):
            debris_x = np.random.random() * width
            debris_y = np.random.random() * height
            debris_size = 2 + np.random.random() * 5
            draw.ellipse([debris_x, debris_y, debris_x + debris_size, debris_y + debris_size], 
                         fill=(150, 150, 150, 128))  # Gray debris
        
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
            st.experimental_rerun()
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
