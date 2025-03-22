"""Interactive visualization for SAM samples using Plotly.

This functionality was partly generated using LLMs.
"""


import matplotlib.colors as mcolors
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from nnx.data.data_structures import SAMSample


def visualize_sam_sample(
    sample: SAMSample,
    figsize: tuple[int, int] = (1000, 1000),
    mask_opacity: float = 0.3,
    contour_width: int = 1,
    point_size: int = 12,
    colormap: list[str] | None = None,
) -> go.Figure:
    """Create an interactive Plotly figure to visualize a SAM sample.

    Args:
        sample: The SAMSample to visualize
        figsize: Width and height of the figure in pixels
        mask_opacity: Opacity of mask overlays (0-1)
        contour_width: Width of contour lines
        point_size: Size of point markers
        colormap: List of colors for masks (defaults to distinguishable colors)

    Returns:
        Plotly figure with interactive visualization

    """
    # Get image dimensions
    img_height, img_width = sample.image.shape[:2]

    # Handle grayscale vs RGB
    img = sample.image
    if len(img.shape) == 2:
        # Convert grayscale to RGB for visualization
        img = np.stack([img, img, img], axis=2)
    elif len(img.shape) == 3 and img.shape[2] == 1:
        # Convert single-channel to RGB
        img = np.concatenate([img] * 3, axis=2)

    # Scale image to [0, 255] for proper display
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)

    # Create figure - using subplots for more control
    fig = make_subplots(rows=1, cols=1)

    # Add base image
    fig.add_trace(
        go.Image(
            z=img,
            name="Base Image",
        ),
    )

    # Generate colors for masks if not provided
    if colormap is None:
        # Use distinguishable colors
        colors = list(mcolors.TABLEAU_COLORS.values())
        colors.extend(list(mcolors.CSS4_COLORS.values()))
        # Ensure we have enough colors
        colormap = [colors[i % len(colors)] for i in range(len(sample.bitmasks))]

    # Track legend groups for button controls
    legend_groups = ["all_masks", "all_contours", "all_points"]

    # Add each mask as both filled area and contour
    for i, mask in enumerate(sample.bitmasks):
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
                legendgrouptitle=dict(text=mask_name),
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

    # Add points grouped by bitmask
    if sample.points:
        for i, point_group in enumerate(sample.points):
            if not point_group:
                continue

            mask_name = f"Mask {i + 1}"
            legend_group = f"mask_{i}"

            # Group points by whether they're positive or negative
            pos_points = [p for p in point_group if p.positive]
            neg_points = [p for p in point_group if not p.positive]

            # Add positive points - NOTE: x and y are flipped as required
            if pos_points:
                # Convert normalized coordinates to pixel coordinates
                # Flip x and y as mentioned in the requirements
                pos_y = [int(p.x * p.height) for p in pos_points]
                pos_x = [int(p.y * p.width) for p in pos_points]

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

            # Add negative points - NOTE: x and y are flipped as required
            if neg_points:
                # Convert normalized coordinates to pixel coordinates
                # Flip x and y as mentioned in the requirements
                neg_y = [int(p.x * p.height) for p in neg_points]
                neg_x = [int(p.y * p.width) for p in neg_points]

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
                "y": 1.01,
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
