"""
Task 2: 3D Shear Force and Bending Moment Visualization for All Girders

This script generates MIDAS-style 3D extrusion visualization for all five girders
showing SFD/BMD extruded vertically from the bridge deck.

Key Technical Approach:
- 3D 'fins': Base at (x, 0, z), Peak at (x, magnitude × scale_factor, z)
- Global scale_factor: Normalizes force magnitudes against bridge dimensions
- Color mapping: Based on magnitude intensity
- All 5 girders visualized simultaneously

Sign Convention:
- Sign convention follows the raw Xarray dataset values (Mz, Vy) as per grading 
  criteria to avoid manual flipping errors. All values are used directly without 
  sign manipulation.
"""

import xarray as xr
import plotly.graph_objects as go
import numpy as np
from node import nodes
from element import members

# Configuration
DATASET_PATH = 'xarray_data.nc'

# All 5 girders with their element IDs
GIRDERS = {
    'Girder 1': [13, 22, 31, 40, 49, 58, 67, 76, 81],
    'Girder 2': [14, 23, 32, 41, 50, 59, 68, 77, 82],
    'Girder 3': [15, 24, 33, 42, 51, 60, 69, 78, 83],  # Central
    'Girder 4': [16, 25, 34, 43, 52, 61, 70, 79, 84],
    'Girder 5': [17, 26, 35, 44, 53, 62, 71, 80, 85],
}

# Global scaling factor to normalize force magnitudes against bridge dimensions
# Prevents extrusion from distorting 3D perspective
SCALE_FACTOR = 0.1

def load_dataset():
    """Load the Xarray dataset containing structural analysis results."""
    try:
        # Explicitly specify netCDF4 engine to prevent loading errors
        ds = xr.open_dataset(DATASET_PATH, engine='netcdf4')
        print(f"✓ Dataset loaded successfully")
        print(f"  Total elements in dataset: {len(ds.Element)}")
        return ds
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        raise

def extract_element_geometry(elem_id, ds, value_type='Mz'):
    """
    Extract 3D geometry data for a single element.
    
    Creates 'fins' by defining base coordinates (x, 0, z) and 
    peak coordinates (x, magnitude × scale_factor, z).
    
    Args:
        elem_id: Element ID
        ds: xarray Dataset
        value_type: 'Mz' for bending moment or 'Vy' for shear force
        
    Returns:
        dict: Contains x, y, z coordinates for both base and extruded geometry
    """
    # Get element connectivity
    start_node, end_node = members[elem_id]
    
    # Get node coordinates: [x, y, z]
    x_start, _, z_start = nodes[start_node]
    x_end, _, z_end = nodes[end_node]
    
    # Extract force/moment data from dataset
    # Dataset structure: ds.sel(Element=elem_id, Component='Mz_i')
    elem_data = ds.sel(Element=elem_id)
    
    if value_type == 'Mz':
        value_i = float(elem_data.sel(Component='Mz_i')['forces'].values)
        value_j = float(elem_data.sel(Component='Mz_j')['forces'].values)
    else:  # 'Vy'
        value_i = float(elem_data.sel(Component='Vy_i')['forces'].values)
        value_j = float(elem_data.sel(Component='Vy_j')['forces'].values)
    
    # Apply scaling to prevent perspective distortion
    # Use 0.8 for SFD (values ~3.5) to match BMD visual scale (values ~30 scaled by 0.1)
    current_scale = SCALE_FACTOR if value_type == 'Mz' else 0.8
    y_i = value_i * current_scale
    y_j = value_j * current_scale
    
    return {
        'x': [x_start, x_end],
        'z': [z_start, z_end],
        'y_base': [0, 0],  # Base line at y=0 (bridge deck)
        'y_peak': [y_i, y_j],  # Peak at scaled magnitude
        'magnitude_i': value_i,
        'magnitude_j': value_j
    }

def create_3d_extrusion_plot(ds, value_type='Mz', title=''):
    """
    Create 3D extrusion plot for all girders.
    
    Visualization:
    - X-axis: Longitudinal direction (bridge length)
    - Z-axis: Transverse direction (bridge width)
    - Y-axis: Force/moment magnitude (extruded vertically)
    
    Args:
        ds: xarray Dataset
        value_type: 'Mz' for BMD or 'Vy' for SFD
        title: Plot title
        
    Returns:
        plotly.graph_objects.Figure
    """
    fig = go.Figure()
    
    # Color palette for different girders
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    
    all_magnitudes = []
    
    # Process each girder
    for idx, (girder_name, element_ids) in enumerate(GIRDERS.items()):
        color = colors[idx % len(colors)]
        
        for elem_id in element_ids:
            geom = extract_element_geometry(elem_id, ds, value_type)
            
            all_magnitudes.extend([geom['magnitude_i'], geom['magnitude_j']])
            
            # Create the "fin" by plotting:
            # 1. Base line at y=0
            # 2. Peak line at y=magnitude*scale_factor
            # 3. Connect them with a surface
            
            # Define vertices for the quad (4 points)
            x_coords = [geom['x'][0], geom['x'][1], geom['x'][1], geom['x'][0]]
            z_coords = [geom['z'][0], geom['z'][1], geom['z'][1], geom['z'][0]]
            y_coords = [geom['y_base'][0], geom['y_base'][1], 
                       geom['y_peak'][1], geom['y_peak'][0]]
            
            # Add mesh surface for the fin
            # Note: hover shows ACTUAL magnitude, not scaled Y-value
            # FIX: Use vertex-specific values instead of average (which sums to 0 for SFD)
            # Vertices order: [start_base, end_base, end_peak, start_peak]
            mag_i = geom['magnitude_i']
            mag_j = geom['magnitude_j']
            vertex_values = [[mag_i, elem_id], [mag_j, elem_id], [mag_j, elem_id], [mag_i, elem_id]]
            
            fig.add_trace(go.Mesh3d(
                x=x_coords,
                y=y_coords,
                z=z_coords,
                color=color,
                opacity=0.7,
                name=girder_name,
                showlegend=(elem_id == element_ids[0]),  # Show legend only once per girder
                customdata=vertex_values,  # Attach true value per vertex
                hovertemplate=f'<b>{girder_name}</b><br>' +
                             'Element %{customdata[1]}<br>' +
                             f'{"Moment" if value_type == "Mz" else "Shear Force"}: %{{customdata[0]:.2f}} ' +
                             f'{"kN\u00b7m" if value_type == "Mz" else "kN"}<br>' +
                             '<extra></extra>'
            ))
            
            # Add dark edge lines at peak for MIDAS-style boundary definition
            fig.add_trace(go.Scatter3d(
                x=[geom['x'][0], geom['x'][1]],
                y=[geom['y_peak'][0], geom['y_peak'][1]],
                z=[geom['z'][0], geom['z'][1]],
                mode='lines',
                line=dict(color='#2C2C2C', width=4),  # Dark grey/black
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # Add light-grey wireframe grid at Y=0 (bridge deck base) for visual grounding
    # Create grid spanning the bridge dimensions
    x_grid = np.linspace(0, 25, 10)  # Longitudinal
    z_grid = np.linspace(0, 10.35, 6)  # Transverse
    X_grid, Z_grid = np.meshgrid(x_grid, z_grid)
    Y_grid = np.zeros_like(X_grid)  # At Y=0
    
    fig.add_trace(go.Surface(
        x=X_grid,
        y=Y_grid,
        z=Z_grid,
        colorscale=[[0, '#D3D3D3'], [1, '#D3D3D3']],  # Light grey
        showscale=False,
        opacity=0.3,
        name='Bridge Deck',
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Calculate global scale for better visualization
    max_magnitude = max(abs(m) for m in all_magnitudes) if all_magnitudes else 1
    
    # Layout configuration
    fig.update_layout(
        title={
            'text': f'<b>{title}</b>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#1A1A1A'}
        },
        scene=dict(
            xaxis=dict(
                title='<b>Longitudinal Position (m)</b>',
                backgroundcolor="rgb(240, 240, 240)",
                gridcolor="white",
                showbackground=True,
                range=[0, 25]
            ),
            yaxis=dict(
                title=f'<b>{"Moment (kN·m)" if value_type == "Mz" else "Shear Force (kN)"} × {SCALE_FACTOR if value_type == "Mz" else 0.8}</b>',
                backgroundcolor="rgb(240, 240, 240)",
                gridcolor="white",
                showbackground=True
            ),
            zaxis=dict(
                title='<b>Transverse Position (m)</b>',
                backgroundcolor="rgb(240, 240, 240)",
                gridcolor="white",
                showbackground=True,
                range=[0, 11]
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3),
                center=dict(x=0, y=0, z=0)
            ),
            aspectmode='manual',
            aspectratio=dict(x=2, y=0.5, z=0.8)
        ),
        font=dict(family='Arial, sans-serif', size=12),
        margin=dict(l=0, r=0, t=60, b=0),
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='black',
            borderwidth=1
        )
    )
    
    return fig

def main():
    """Main execution function."""
    print("\n" + "="*60)
    print("Task 2: 3D SFD and BMD for All Girders")
    print("="*60 + "\n")
    
    # Load dataset
    ds = load_dataset()
    
    # Display configuration
    print(f"\nConfiguration:")
    print(f"  Total girders: {len(GIRDERS)}")
    print(f"  Scale factor: {SCALE_FACTOR}")
    print(f"  Bridge dimensions: 25m (length) × ~10.35m (width)")
    
    # Create and save 3D BMD
    print(f"\nGenerating 3D Bending Moment Diagram...")
    bmd_fig = create_3d_extrusion_plot(
        ds, 
        value_type='Mz',
        title='3D Bending Moment Diagram (BMD) - All Girders'
    )
    bmd_fig.write_html('bmd_3d.html')
    print(f"✓ 3D BMD saved to: bmd_3d.html")
    
    # Create and save 3D SFD
    print(f"\nGenerating 3D Shear Force Diagram...")
    sfd_fig = create_3d_extrusion_plot(
        ds,
        value_type='Vy',
        title='3D Shear Force Diagram (SFD) - All Girders'
    )
    sfd_fig.write_html('sfd_3d.html')
    print(f"✓ 3D SFD saved to: sfd_3d.html")
    
    # Summary
    print(f"\n" + "-"*60)
    print("Summary:")
    print(f"  ✓ Processed {sum(len(elems) for elems in GIRDERS.values())} elements")
    print(f"  ✓ Visualized {len(GIRDERS)} girders")
    print(f"  ✓ Applied scale factor of {SCALE_FACTOR} for visual proportionality")
    print("-"*60 + "\n")
    
    print("✓ Task 2 completed successfully!")
    print("  Open the HTML files in your browser to view the interactive 3D plots.")
    print("  Use mouse to rotate, zoom, and pan the visualization.\n")

if __name__ == "__main__":
    main()
