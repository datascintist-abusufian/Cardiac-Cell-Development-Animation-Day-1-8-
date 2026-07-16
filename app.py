import streamlit as st
import time
import pandas as pd
import numpy as np
import altair as alt
from PIL import Image, ImageDraw
import io
import base64
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Cardiac Cell Development Advanced Analysis", layout="wide")

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
def initialize_session_state():
    """Initialize all session state variables"""
    if 'play_animation' not in st.session_state:
        st.session_state.play_animation = False
    if 'last_day' not in st.session_state:
        st.session_state.last_day = 1
    if 'animation_speed' not in st.session_state:
        st.session_state.animation_speed = 2
    if 'current_day' not in st.session_state:
        st.session_state.current_day = 1
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
    if 'future_days' not in st.session_state:
        st.session_state.future_days = None

# Initialize session state
initialize_session_state()

# ============================================================================
# TITLE AND DESCRIPTION
# ============================================================================
st.title("🧬 Cardiac Cell Development Advanced Analysis (Day 1-8)")
st.markdown("Comprehensive visualization and analysis of cardiac cell morphological and functional changes over an 8-day period with predictive modeling")

# ============================================================================
# CELL DATA
# ============================================================================
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
        "color": (255, 214, 204),
        "beat": 0.2,
        "sync_level": 0.1,
        "cell_count": 8,
        "debris_level": 0.1,
        "sarcomere_organization": 0.1,
        "mitochondrial_density": 0.2,
        "membrane_integrity": 0.9,
        "nuclear_condensation": 0.3,
        "apoptotic_markers": 0.05,
        "metabolic_activity": 0.3,
        "calcium_handling": 0.1,
        "gap_junction": 0.05,
        "stress_markers": 0.1
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
        "color": (255, 204, 204),
        "beat": 0.4,
        "sync_level": 0.2,
        "cell_count": 12,
        "debris_level": 0.1,
        "sarcomere_organization": 0.25,
        "mitochondrial_density": 0.3,
        "membrane_integrity": 0.85,
        "nuclear_condensation": 0.25,
        "apoptotic_markers": 0.08,
        "metabolic_activity": 0.4,
        "calcium_handling": 0.2,
        "gap_junction": 0.15,
        "stress_markers": 0.15
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
        "color": (255, 204, 204),
        "beat": 0.5,
        "sync_level": 0.3,
        "cell_count": 16,
        "debris_level": 0.2,
        "sarcomere_organization": 0.4,
        "mitochondrial_density": 0.4,
        "membrane_integrity": 0.8,
        "nuclear_condensation": 0.2,
        "apoptotic_markers": 0.1,
        "metabolic_activity": 0.5,
        "calcium_handling": 0.35,
        "gap_junction": 0.3,
        "stress_markers": 0.2
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
        "color": (255, 153, 153),
        "beat": 0.7,
        "sync_level": 0.5,
        "cell_count": 20,
        "debris_level": 0.2,
        "sarcomere_organization": 0.6,
        "mitochondrial_density": 0.55,
        "membrane_integrity": 0.75,
        "nuclear_condensation": 0.15,
        "apoptotic_markers": 0.12,
        "metabolic_activity": 0.65,
        "calcium_handling": 0.5,
        "gap_junction": 0.5,
        "stress_markers": 0.25
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
        "color": (255, 102, 102),
        "beat": 0.8,
        "sync_level": 0.7,
        "cell_count": 24,
        "debris_level": 0.3,
        "sarcomere_organization": 0.75,
        "mitochondrial_density": 0.7,
        "membrane_integrity": 0.7,
        "nuclear_condensation": 0.1,
        "apoptotic_markers": 0.15,
        "metabolic_activity": 0.8,
        "calcium_handling": 0.7,
        "gap_junction": 0.7,
        "stress_markers": 0.3
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
        "color": (255, 51, 51),
        "beat": 1.0,
        "sync_level": 0.9,
        "cell_count": 28,
        "debris_level": 0.4,
        "sarcomere_organization": 0.9,
        "mitochondrial_density": 0.85,
        "membrane_integrity": 0.65,
        "nuclear_condensation": 0.1,
        "apoptotic_markers": 0.2,
        "metabolic_activity": 0.9,
        "calcium_handling": 0.85,
        "gap_junction": 0.85,
        "stress_markers": 0.4
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
        "color": (204, 51, 51),
        "beat": 0.6,
        "sync_level": 0.5,
        "cell_count": 20,
        "debris_level": 0.7,
        "sarcomere_organization": 0.5,
        "mitochondrial_density": 0.5,
        "membrane_integrity": 0.4,
        "nuclear_condensation": 0.5,
        "apoptotic_markers": 0.5,
        "metabolic_activity": 0.5,
        "calcium_handling": 0.4,
        "gap_junction": 0.4,
        "stress_markers": 0.7
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
        "color": (153, 51, 51),
        "beat": 0.2,
        "sync_level": 0.2,
        "cell_count": 12,
        "debris_level": 0.9,
        "sarcomere_organization": 0.2,
        "mitochondrial_density": 0.2,
        "membrane_integrity": 0.2,
        "nuclear_condensation": 0.8,
        "apoptotic_markers": 0.8,
        "metabolic_activity": 0.2,
        "calcium_handling": 0.15,
        "gap_junction": 0.1,
        "stress_markers": 0.9
    }
}

# ============================================================================
# ANALYTICS FUNCTIONS
# ============================================================================
@st.cache_data
def prepare_dataset():
    """Prepare dataset for machine learning analysis"""
    data = []
    for day in range(1, 9):
        d = cell_data[day]
        features = {
            'day': day,
            'beat': d['beat'],
            'sync_level': d['sync_level'],
            'cell_count': d['cell_count'],
            'debris_level': d['debris_level'],
            'sarcomere_organization': d['sarcomere_organization'],
            'mitochondrial_density': d['mitochondrial_density'],
            'membrane_integrity': d['membrane_integrity'],
            'nuclear_condensation': d['nuclear_condensation'],
            'apoptotic_markers': d['apoptotic_markers'],
            'metabolic_activity': d['metabolic_activity'],
            'calcium_handling': d['calcium_handling'],
            'gap_junction': d['gap_junction'],
            'stress_markers': d['stress_markers']
        }
        data.append(features)
    return pd.DataFrame(data)

@st.cache_data
def train_predictive_models(df):
    """Train predictive models for various outcomes"""
    models = {}
    
    # Target variables to predict
    targets = ['beat', 'sync_level', 'cell_count', 'debris_level']
    
    for target in targets:
        X = df[['day']].values
        y = df[target].values
        
        # Use Random Forest for prediction
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        models[target] = model
    
    return models

def predict_future(models, days_ahead=5):
    """Predict future values for up to 5 days ahead"""
    predictions = {}
    future_days = list(range(9, 9 + days_ahead))
    
    for target, model in models.items():
        X_future = np.array(future_days).reshape(-1, 1)
        preds = model.predict(X_future)
        predictions[target] = dict(zip(future_days, preds))
    
    return predictions, future_days

def generate_cell_frame(day_num, pulse=0.0):
    """Generate cell animation frame with realistic morphology"""
    day_data = cell_data[day_num]
    width, height = 800, 600
    
    image = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    
    # Create cell clusters
    num_clusters = max(3, day_num)
    cluster_centers = []
    np.random.seed(42)  # For reproducibility
    for _ in range(num_clusters):
        cluster_centers.append((50 + np.random.random() * 700, 50 + np.random.random() * 500))
    
    # Draw connecting fibers (days 4-6)
    if 4 <= day_num <= 6:
        for i in range(len(cluster_centers)):
            for j in range(i+1, len(cluster_centers)):
                if np.random.random() < 0.6:
                    x1, y1 = cluster_centers[i]
                    x2, y2 = cluster_centers[j]
                    fiber_color = (255, 180, 180, 100)
                    points = []
                    steps = 10
                    for step in range(steps + 1):
                        t = step / steps
                        x = x1 + (x2 - x1) * t
                        y = y1 + (y2 - y1) * t
                        if 0 < step < steps:
                            wave_amp = 10 + 5 * np.sin(step)
                            x += np.sin(step * 3) * wave_amp
                            y += np.cos(step * 2) * wave_amp
                        points.append((x, y))
                    
                    for k in range(len(points) - 1):
                        draw.line([points[k], points[k+1]], fill=fiber_color, width=2)
    
    # Draw cells
    cells_drawn = 0
    clusters_used = 0
    
    while cells_drawn < day_data["cell_count"]:
        if clusters_used < len(cluster_centers):
            cluster_x, cluster_y = cluster_centers[clusters_used]
            clusters_used += 1
        else:
            cluster_index = np.random.randint(0, len(cluster_centers))
            cluster_x, cluster_y = cluster_centers[cluster_index]
        
        cells_in_cluster = min(
            max(2, int(day_data["cell_count"] / num_clusters + np.random.randint(-2, 3))),
            day_data["cell_count"] - cells_drawn
        )
        
        for _ in range(cells_in_cluster):
            cluster_radius = 30 + day_num * 5
            angle = np.random.random() * 2 * np.pi
            distance = np.random.random() * cluster_radius
            x = cluster_x + np.cos(angle) * distance
            y = cluster_y + np.sin(angle) * distance
            
            # Cell size based on day
            cell_width = 15 + np.random.random() * 10
            cell_height = cell_width * (1.0 + np.random.random() * (day_num / 4))
            
            # Beating effect
            if np.random.random() < day_data["beat"]:
                beat_factor = 1.0 + pulse * day_data["beat"] * 0.2
                cell_width *= beat_factor
                cell_height *= beat_factor
            
            # Cell color based on health
            health_factor = 1.0 - (day_data["debris_level"] * 0.3)
            r = int(255 * health_factor * (0.6 + 0.4 * (day_num / 8)))
            g = int(200 * health_factor * (1 - day_num / 12))
            b = int(200 * health_factor * (1 - day_num / 12))
            cell_color = (r, g, b, 180)
            
            draw.ellipse([x, y, x + cell_width, y + cell_height], fill=cell_color)
            
            # Draw nucleus
            nucleus_size = max(0.2, 0.7 - day_num * 0.05)
            nucleus_color = (102, 102, 204, 200)
            nucleus_x = x + (cell_width / 2) - (cell_width * nucleus_size / 2)
            nucleus_y = y + (cell_height / 2) - (cell_height * nucleus_size / 2)
            nucleus_width = cell_width * nucleus_size
            nucleus_height = cell_height * nucleus_size
            draw.ellipse([nucleus_x, nucleus_y, nucleus_x + nucleus_width, nucleus_y + nucleus_height], 
                         fill=nucleus_color)
            
            cells_drawn += 1
    
    # Add debris
    base_debris = int(day_data["debris_level"] * 100)
    for i in range(base_debris):
        debris_x = np.random.random() * width
        debris_y = np.random.random() * height
        debris_size = 2 + np.random.random() * 4
        debris_color = (160, 100, 100, 120)
        draw.ellipse([debris_x, debris_y, debris_x + debris_size, debris_y + debris_size], 
                     fill=debris_color)
    
    return image

# ============================================================================
# MAIN APP
# ============================================================================
# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎬 Animation", 
    "📊 Cell Properties", 
    "📈 Data Visualization",
    "🤖 Predictive Analysis",
    "🔬 Advanced Analytics"
])

# ============================================================================
# TAB 1: ANIMATION
# ============================================================================
with tab1:
    # Animation controls
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        auto_play = st.checkbox("Auto Play", value=st.session_state.play_animation)
        st.session_state.play_animation = auto_play
    
    with col2:
        day = st.slider("Select Day", min_value=1, max_value=8, value=st.session_state.current_day)
        st.session_state.current_day = day
    
    with col3:
        speed_options = {1: "Slow", 2: "Medium", 3: "Fast"}
        speed = st.selectbox("Animation Speed", options=[1, 2, 3], 
                           format_func=lambda x: speed_options[x],
                           index=1)
        frame_delay = 1.0 / speed
    
    # Display current day
    current_day_data = cell_data[day]
    st.subheader(current_day_data["title"])
    
    # Animation placeholder
    animation_placeholder = st.empty()
    
    # Auto-play logic
    if auto_play:
        for pulse in np.linspace(0, 1, 10):
            if not st.session_state.play_animation:
                break
                
            frame = generate_cell_frame(day, pulse)
            buf = io.BytesIO()
            frame.save(buf, format="PNG")
            byte_im = buf.getvalue()
            animation_placeholder.image(byte_im, use_container_width=True)
            time.sleep(frame_delay/10)
            
        if st.session_state.play_animation:
            next_day = day + 1 if day < 8 else 1
            st.session_state.current_day = next_day
            time.sleep(frame_delay)
            st.rerun()
    else:
        frame = generate_cell_frame(day, 0.5)
        buf = io.BytesIO()
        frame.save(buf, format="PNG")
        byte_im = buf.getvalue()
        animation_placeholder.image(byte_im, use_container_width=True)

# ============================================================================
# TAB 2: CELL PROPERTIES
# ============================================================================
with tab2:
    st.subheader(f"Cell Properties - {cell_data[day]['title']}")
    
    col1, col2, col3 = st.columns(3)
    
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
    
    with col3:
        st.markdown("### Molecular Markers")
        markers = {
            'Sarcomere Organization': cell_data[day]['sarcomere_organization'],
            'Mitochondrial Density': cell_data[day]['mitochondrial_density'],
            'Membrane Integrity': cell_data[day]['membrane_integrity'],
            'Nuclear Condensation': cell_data[day]['nuclear_condensation'],
            'Apoptotic Markers': cell_data[day]['apoptotic_markers'],
            'Metabolic Activity': cell_data[day]['metabolic_activity'],
            'Calcium Handling': cell_data[day]['calcium_handling'],
            'Gap Junction': cell_data[day]['gap_junction'],
            'Stress Markers': cell_data[day]['stress_markers']
        }
        
        for marker, value in markers.items():
            if value >= 0.7:
                status = "🟢 High"
            elif value >= 0.4:
                status = "🟡 Moderate"
            else:
                status = "🔴 Low"
            st.markdown(f"**{marker}:** {status} ({value:.2f})")

# ============================================================================
# TAB 3: DATA VISUALIZATION
# ============================================================================
with tab3:
    st.subheader("Quantitative Changes Over Time")
    
    # Prepare data
    df = prepare_dataset()
    
    # Metric selection
    all_metrics = ['beat', 'sync_level', 'cell_count', 'debris_level', 
                   'sarcomere_organization', 'mitochondrial_density', 
                   'membrane_integrity', 'nuclear_condensation', 
                   'apoptotic_markers', 'metabolic_activity', 
                   'calcium_handling', 'gap_junction', 'stress_markers']
    
    metric_labels = {
        'beat': 'Beat Strength',
        'sync_level': 'Synchronization',
        'cell_count': 'Cell Count',
        'debris_level': 'Debris Level',
        'sarcomere_organization': 'Sarcomere Organization',
        'mitochondrial_density': 'Mitochondrial Density',
        'membrane_integrity': 'Membrane Integrity',
        'nuclear_condensation': 'Nuclear Condensation',
        'apoptotic_markers': 'Apoptotic Markers',
        'metabolic_activity': 'Metabolic Activity',
        'calcium_handling': 'Calcium Handling',
        'gap_junction': 'Gap Junction',
        'stress_markers': 'Stress Markers'
    }
    
    selected_metrics = st.multiselect(
        "Select metrics to display",
        all_metrics,
        default=['beat', 'sync_level', 'cell_count', 'debris_level'],
        format_func=lambda x: metric_labels[x]
    )
    
    if selected_metrics:
        # Create long-form dataframe
        df_long = pd.melt(
            df[['day'] + selected_metrics], 
            id_vars=['day'], 
            value_vars=selected_metrics,
            var_name='Metric', 
            value_name='Value'
        )
        
        # Replace metric names with labels
        df_long['Metric'] = df_long['Metric'].map(metric_labels)
        
        # Create the chart
        chart = alt.Chart(df_long).mark_line(point=True).encode(
            x=alt.X("day:O", title="Day"),
            y=alt.Y("Value:Q", title="Value (Normalized)"),
            color=alt.Color("Metric:N", title="Metric"),
            tooltip=["day", "Metric", "Value"]
        ).properties(
            width=700,
            height=400,
            title="Cell Development Metrics Over Time"
        ).interactive()
        
        st.altair_chart(chart, use_container_width=True)
        
        # Heatmap
        st.subheader("Feature Correlation Heatmap")
        corr_matrix = df[selected_metrics].corr()
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Correlation Between Features",
            color_continuous_scale="RdBu_r"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        with st.expander("Show Data Table"):
            display_df = df[['day'] + selected_metrics].copy()
            display_df.columns = ['Day'] + [metric_labels[col] for col in selected_metrics]
            st.dataframe(display_df)

# ============================================================================
# TAB 4: PREDICTIVE ANALYSIS
# ============================================================================
with tab4:
    st.subheader("🤖 Predictive Analysis")
    st.markdown("Machine learning models predicting future cell behavior")
    
    # Train models
    df = prepare_dataset()
    models = train_predictive_models(df)
    
    # Predict future
    days_ahead = st.slider("Days to predict ahead", 1, 10, 5)
    predictions, future_days = predict_future(models, days_ahead)
    
    # Display predictions
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Predicted Trends")
        
        # Create prediction plot
        fig = make_subplots(rows=2, cols=2, subplot_titles=['Beat Strength', 'Synchronization', 'Cell Count', 'Debris Level'])
        
        targets = ['beat', 'sync_level', 'cell_count', 'debris_level']
        target_labels = ['Beat Strength', 'Synchronization', 'Cell Count', 'Debris Level']
        
        for idx, (target, label) in enumerate(zip(targets, target_labels)):
            row = idx // 2 + 1
            col = idx % 2 + 1
            
            # Historical data
            historical = df[['day', target]].copy()
            historical['Type'] = 'Historical'
            
            # Predicted data
            pred_data = []
            for day_val, value in predictions[target].items():
                pred_data.append({'day': day_val, target: value, 'Type': 'Predicted'})
            pred_df = pd.DataFrame(pred_data)
            
            # Combine
            combined = pd.concat([historical, pred_df], ignore_index=True)
            
            fig.add_trace(
                go.Scatter(
                    x=combined['day'], 
                    y=combined[target],
                    mode='lines+markers',
                    name=label,
                    line=dict(color='blue' if target in ['beat', 'sync_level'] else 'red'),
                    legendgroup=label
                ),
                row=row, col=col
            )
            
            # Add confidence intervals (simplified)
            if target in predictions:
                pred_values = np.array([predictions[target][d] for d in predictions[target].keys()])
                std_dev = 0.1 * pred_values
                upper = [predictions[target][d] + std_dev[i] for i, d in enumerate(predictions[target].keys())]
                lower = [predictions[target][d] - std_dev[i] for i, d in enumerate(predictions[target].keys())]
                
                fig.add_trace(
                    go.Scatter(
                        x=list(predictions[target].keys()),
                        y=upper,
                        mode='lines',
                        fill=None,
                        line=dict(width=0),
                        showlegend=False,
                        legendgroup=label
                    ),
                    row=row, col=col
                )
                fig.add_trace(
                    go.Scatter(
                        x=list(predictions[target].keys()),
                        y=lower,
                        mode='lines',
                        fill='tonexty',
                        line=dict(width=0),
                        showlegend=False,
                        legendgroup=label
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(height=600, title_text="Predictions vs Historical Data")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Model Performance")
        
        # Feature importance
        feature_importance = models['beat'].feature_importances_
        importance_df = pd.DataFrame({
            'Feature': ['Day'],
            'Importance': feature_importance
        }).sort_values('Importance', ascending=True)
        
        fig_importance = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Feature Importance for Beat Prediction',
            color='Importance',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_importance, use_container_width=True)
        
        # Prediction table
        st.markdown("### Predicted Values")
        pred_df = pd.DataFrame(predictions)
        pred_df.index = future_days
        pred_df.columns = [metric_labels[col] for col in pred_df.columns]
        st.dataframe(pred_df)

# ============================================================================
# TAB 5: ADVANCED ANALYTICS
# ============================================================================
with tab5:
    st.subheader("🔬 Advanced Analytics")
    
    # Prepare data
    df = prepare_dataset()
    
    # PCA Analysis
    st.markdown("### PCA Clustering Analysis")
    
    features = ['beat', 'sync_level', 'cell_count', 'debris_level', 
                'sarcomere_organization', 'mitochondrial_density', 
                'membrane_integrity', 'nuclear_condensation', 
                'apoptotic_markers', 'metabolic_activity', 
                'calcium_handling', 'gap_junction', 'stress_markers']
    
    # Scale features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])
    
    # Apply PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)
    
    # Create PCA plot
    pca_df = pd.DataFrame({
        'PC1': pca_result[:, 0],
        'PC2': pca_result[:, 1],
        'Day': df['day'],
        'Cluster': clusters,
        'Beat': df['beat'],
        'Sync': df['sync_level']
    })
    
    fig_pca = px.scatter(
        pca_df,
        x='PC1',
        y='PC2',
        color='Cluster',
        size='Beat',
        hover_data=['Day', 'Sync'],
        title='PCA Clustering of Cell States',
        labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)',
                'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)'}
    )
    
    # Add day annotations
    for i, row in pca_df.iterrows():
        fig_pca.add_annotation(
            x=row['PC1'],
            y=row['PC2'],
            text=str(row['Day']),
            showarrow=True,
            arrowhead=1
        )
    
    st.plotly_chart(fig_pca, use_container_width=True)
    
    # Explained variance
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Explained Variance")
        var_df = pd.DataFrame({
            'Principal Component': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
            'Explained Variance': pca.explained_variance_ratio_
        })
        
        fig_var = px.bar(
            var_df,
            x='Principal Component',
            y='Explained Variance',
            title='PCA Explained Variance',
            color='Explained Variance',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_var, use_container_width=True)
    
    with col2:
        st.markdown("### Cluster Statistics")
        cluster_stats = pca_df.groupby('Cluster').agg({
            'Beat': ['mean', 'std'],
            'Sync': ['mean', 'std'],
            'Day': ['min', 'max']
        }).round(3)
        st.dataframe(cluster_stats)
    
    # Overall health score
    st.markdown("### Cell Health Score Analysis")
    
    # Calculate health score
    health_weights = {
        'sarcomere_organization': 0.20,
        'mitochondrial_density': 0.15,
        'membrane_integrity': 0.15,
        'metabolic_activity': 0.15,
        'calcium_handling': 0.15,
        'gap_junction': 0.10,
        'nuclear_condensation': -0.10,
        'apoptotic_markers': -0.10,
        'stress_markers': -0.10
    }
    
    df['health_score'] = 0
    for feature, weight in health_weights.items():
        df['health_score'] += df[feature] * weight
    
    # Normalize to 0-100
    df['health_score'] = (df['health_score'] - df['health_score'].min()) / (df['health_score'].max() - df['health_score'].min()) * 100
    
    # Create health score plot
    fig_health = go.Figure()
    
    fig_health.add_trace(go.Scatter(
        x=df['day'],
        y=df['health_score'],
        mode='lines+markers',
        name='Cell Health Score',
        line=dict(color='green', width=3),
        marker=dict(size=10)
    ))
    
    # Add threshold lines
    fig_health.add_hline(y=70, line_dash="dash", line_color="green", annotation_text="Healthy Threshold")
    fig_health.add_hline(y=50, line_dash="dash", line_color="orange", annotation_text="Moderate Threshold")
    fig_health.add_hline(y=30, line_dash="dash", line_color="red", annotation_text="Critical Threshold")
    
    fig_health.update_layout(
        title="Cell Health Score Over Time (0-100)",
        xaxis_title="Day",
        yaxis_title="Health Score",
        height=400
    )
    
    st.plotly_chart(fig_health, use_container_width=True)
    
    # Health score table
    with st.expander("Show Health Score Details"):
        health_df = df[['day', 'health_score']].copy()
        health_df['health_status'] = pd.cut(
            health_df['health_score'],
            bins=[0, 30, 50, 70, 100],
            labels=['Critical', 'Moderate', 'Good', 'Excellent']
        )
        st.dataframe(health_df)

# ============================================================================
# SIDEBAR
# ============================================================================
st.sidebar.header("🎮 Controls")
st.sidebar.markdown("""
This app visualizes the development of cardiac cells from Day 1 to Day 8.

### Features:
- **Animation:** Visual representation of cells
- **Cell Properties:** Detailed information about each day
- **Data Visualization:** Quantitative metrics over time
- **Predictive Analysis:** ML models predicting future behavior
- **Advanced Analytics:** PCA clustering and health scoring

### About the data:
Based on a study of cardiac cell development showing morphological and functional changes over 8 days, from immature cells (Day 1) to peak activity (Day 6) and deterioration (Days 7-8).
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔬 Analysis Settings")

# Advanced settings
show_advanced = st.sidebar.checkbox("Show Advanced Settings", value=False)

if show_advanced:
    st.sidebar.markdown("#### Model Parameters")
    n_estimators = st.sidebar.slider("Number of Trees", 50, 300, 100, 50)
    max_depth = st.sidebar.slider("Max Depth", 3, 15, 10, 1)
    
    st.sidebar.markdown("#### Visualization")
    transparency = st.sidebar.slider("Cell Transparency", 0.1, 1.0, 0.8, 0.1)
    cluster_radius = st.sidebar.slider("Cluster Radius", 10, 50, 30, 5)

st.sidebar.markdown("---")
st.sidebar.markdown("📊 **Data Source:** Cardiac Cell Development Study")
st.sidebar.markdown("🔬 **Version:** 2.0 (Advanced Analysis)")
