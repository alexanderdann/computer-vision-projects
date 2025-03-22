"""Interactive visualization for SAM samples using Plotly.

This functionality was partly generated using LLMs.
"""

import matplotlib.colors as mcolors
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from nnx.data.data_structures import SAMSample


def _prepare_image(img: np.ndarray) -> np.ndarray:
    """Prepare image for visualization.

    Args:
        img: Input image array

    Returns:
        Properly formatted RGB image array for visualization

    """
    # Handle grayscale vs RGB
    grays, rgb = 2, 3
    if len(img.shape) == grays:
        # Convert grayscale to RGB for visualization
        img = np.stack([img, img, img], axis=2)
    elif len(img.shape) == rgb and img.shape[2] == 1:
        # Convert single-channel to RGB
        img = np.concatenate([img] * 3, axis=2)

    # Scale image to [0, 255] for proper display
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)

    return img


def _generate_colormap(num_colors: int) -> list[str]:
    """Generate a list of distinguishable colors.

    Args:
        num_colors: Number of distinct colors needed

    Returns:
        List of color strings

    """
    # Use distinguishable colors
    colors = list(mcolors.TABLEAU_COLORS.values())
    colors.extend(list(mcolors.CSS4_COLORS.values()))
    # Ensure we have enough colors
    return [colors[i % len(colors)] for i in range(num_colors)]


def _add_base_image(fig: go.Figure, img: np.ndarray) -> go.Figure:
    """Add base image to the figure.

    Args:
        fig: Plotly figure
        img: Image array

    Returns:
        Updated figure with base image

    """
    fig.add_trace(
        go.Image(
            z=img,
            name="Base Image",
        ),
    )
    return fig


def _add_mask_layers(
    fig: go.Figure,
    masks: list[np.ndarray],
    colormap: list[str],
    mask_opacity: float,
    contour_width: int,
) -> tuple[go.Figure, list[str]]:
    """Add mask layers (filled areas and contours) to the figure.

    Args:
        fig: Plotly figure
        masks: List of binary mask arrays
        colormap: List of colors for each mask
        mask_opacity: Opacity value for mask fills
        contour_width: Width of contour lines

    Returns:
        Updated figure and list of legend groups

    """
    legend_groups = []

    for i, mask in enumerate(masks):
        mask_color = colormap[i % len(colormap)]
        mask_name = f"Mask {i + 1}"
        legend_group = f"mask_{i}"
        legend_groups.append(legend_group)

        # For filled mask areas
        fig.add_trace(
            go.Contour(
                z=mask,
                showscale=False,
                contours={
                    "coloring": "fill",
                    "showlines": False,
                    "start": 0.5,  # Only show where mask > 0.5
                    "end": 1.0,
                    "size": 0.5,
                },
                colorscale=[
                    [0, "rgba(0,0,0,0)"],
                    [1, f"rgba{mcolors.to_rgba(mask_color, alpha=mask_opacity)}"],
                ],
                name=f"{mask_name} (Fill)",
                line={"width": 0},
                hoverinfo="name",
                legendgroup=legend_group,
                legendgrouptitle={"text": mask_name},
            ),
        )

        # For contour lines
        fig.add_trace(
            go.Contour(
                z=mask,
                showscale=False,
                contours={
                    "coloring": "lines",
                    "showlines": True,
                    "start": 0.5,
                    "end": 0.5,
                    "size": 0.1,
                },
                line={"color": mask_color, "width": contour_width},
                name=f"{mask_name} (Contour)",
                hoverinfo="name",
                legendgroup=legend_group,
            ),
        )

    return fig, legend_groups


def _convert_point_coords(points: list, height: int, width: int) -> tuple[list, list]:
    """Convert normalized point coordinates to pixel coordinates.

    Args:
        points: List of point objects
        height: Image height
        width: Image width

    Returns:
        Tuple of lists (x_coords, y_coords)

    """
    # Convert normalized coordinates to pixel coordinates
    # Flip x and y as required
    y_coords = [int(p.x * height) for p in points]
    x_coords = [int(p.y * width) for p in points]

    return x_coords, y_coords


def _add_point_markers(
    fig: go.Figure,
    point_groups: list[list],
    legend_groups: list[str],
    point_size: int,
) -> go.Figure:
    """Add point markers to the figure.

    Args:
        fig: Plotly figure
        point_groups: List of groups of points
        legend_groups: List of legend group names
        point_size: Size of point markers

    Returns:
        Updated figure

    """
    if not point_groups:
        return fig

    for i, point_group in enumerate(point_groups):
        if not point_group:
            continue

        mask_name = f"Mask {i + 1}"
        legend_group = legend_groups[i] if i < len(legend_groups) else f"mask_{i}"

        # Group points by whether they're positive or negative
        pos_points = [p for p in point_group if p.positive]
        neg_points = [p for p in point_group if not p.positive]

        # Sample dimensions from the first point
        if point_group:
            height, width = point_group[0].height, point_group[0].width
        else:
            continue

        # Add positive points
        if pos_points:
            pos_x, pos_y = _convert_point_coords(pos_points, height, width)

            fig.add_trace(
                go.Scatter(
                    x=pos_x,
                    y=pos_y,
                    mode="markers",
                    marker={
                        "size": point_size,
                        "color": "green",
                        "symbol": "star",
                        "line": {"width": 1, "color": "white"},
                    },
                    name=f"{mask_name} (+ Points)",
                    hoverinfo="name",
                    legendgroup=legend_group,
                ),
            )

        # Add negative points
        if neg_points:
            neg_x, neg_y = _convert_point_coords(neg_points, height, width)

            fig.add_trace(
                go.Scatter(
                    x=neg_x,
                    y=neg_y,
                    mode="markers",
                    marker={
                        "size": point_size,
                        "color": "red",
                        "symbol": "x",
                        "line": {"width": 1, "color": "white"},
                    },
                    name=f"{mask_name} (- Points)",
                    hoverinfo="name",
                    legendgroup=legend_group,
                ),
            )

    return fig


def _create_control_buttons(fig: go.Figure) -> list[dict]:
    """Create interactive control buttons for the visualization.

    Args:
        fig: Plotly figure

    Returns:
        List of button configuration dictionaries

    """
    buttons = []

    # Toggle all masks
    buttons.append(
        {
            "label": "Toggle All Masks",
            "method": "update",
            "args": [{"visible": [True] * len(fig.data)}],
            "args2": [{"visible": [i == 0 for i in range(len(fig.data))]}],
        },
    )

    # Toggle contours only
    visible_contours = []
    for i in range(len(fig.data)):
        if i == 0:  # Base image
            visible_contours.append(True)
        else:
            # Only keep contour traces (every 2nd mask trace starting from index 2)
            visible_contours.append(i % 2 == 0 and i > 0)

    buttons.append(
        {
            "label": "Contours Only",
            "method": "update",
            "args": [{"visible": visible_contours}],
        },
    )

    # Toggle points only
    visible_points = [
        i == 0 for i in range(len(fig.data))
    ]  # Start with only base image
    point_indices = []

    # Find indices of point traces
    for i, trace in enumerate(fig.data):
        if hasattr(trace, "mode") and trace.mode == "markers":
            visible_points[i] = True
            point_indices.append(i)

    if point_indices:
        buttons.append(
            {
                "label": "Points Only",
                "method": "update",
                "args": [{"visible": visible_points}],
            },
        )

    # Reset to all visible
    buttons.append(
        {
            "label": "Show All",
            "method": "update",
            "args": [{"visible": [True] * len(fig.data)}],
        },
    )

    return buttons


def _setup_layout(
    fig: go.Figure,
    img_width: int,
    img_height: int,
    figsize: tuple[int, int],
    buttons: list[dict],
) -> go.Figure:
    """Set up the figure layout including axes, controls, and legend.

    Args:
        fig: Plotly figure
        img_width: Width of the image
        img_height: Height of the image
        figsize: Figure size as (width, height)
        buttons: List of control button configurations

    Returns:
        Figure with layout configured

    """
    # Update layout
    fig.update_layout(
        width=figsize[0],
        height=figsize[1],
        title="SAM Sample Visualization",
        xaxis={
            "showgrid": False,
            "zeroline": False,
            "range": [0, img_width],
            "visible": False,
            "scaleanchor": "y",
            "scaleratio": 1,
        },
        yaxis={
            "showgrid": False,
            "zeroline": False,
            "range": [img_height, 0],  # Flip y-axis for image coordinates
            "visible": False,
        },
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
        plot_bgcolor="rgba(0,0,0,0)",
        legend={
            "title": "Toggle Components",
            "orientation": "v",
            "x": 1.02,
            "y": 1,
            "xanchor": "left",
            "bgcolor": "rgba(255,255,255,0.8)",
            "bordercolor": "lightgray",
            "borderwidth": 1,
        },
    )

    fig.update_layout(
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "x": 0.5,
                "y": 0.95,
                "xanchor": "center",
                "yanchor": "bottom",
                "buttons": buttons,
                "pad": {"r": 10, "t": 10},
                "showactive": True,
                "bgcolor": "rgba(255, 255, 255, 0.8)",
                "bordercolor": "lightgray",
            },
        ],
    )

    return fig


def visualize_sam_sample(
    sample: SAMSample,
    figsize: tuple[int, int] = (1000, 1000),
    mask_opacity: float = 0.5,
    contour_width: int = 1,
    point_size: int = 12,
) -> go.Figure:
    """Create an interactive Plotly figure to visualize a SAM sample.

    Args:
        sample: The SAMSample to visualize
        figsize: Width and height of the figure in pixels
        mask_opacity: Opacity of mask overlays (0-1)
        contour_width: Width of contour lines
        point_size: Size of point markers

    Returns:
        Plotly figure with interactive visualization

    """
    # Get image dimensions
    img_height, img_width = sample.image.shape[:2]

    # Prepare the image for visualization
    img = _prepare_image(sample.image)

    # Create figure - using subplots for more control
    fig = make_subplots(rows=1, cols=1)

    # Add base image
    fig = _add_base_image(fig, img)

    # Generate colors for masks
    colormap = _generate_colormap(len(sample.bitmasks))

    # Add mask layers (filled areas and contours)
    fig, legend_groups = _add_mask_layers(
        fig,
        sample.bitmasks,
        colormap,
        mask_opacity,
        contour_width,
    )

    # Add point markers
    fig = _add_point_markers(fig, sample.points, legend_groups, point_size)

    # Create interactive control buttons
    buttons = _create_control_buttons(fig)

    # Set up layout with controls
    return _setup_layout(fig, img_width, img_height, figsize, buttons)
