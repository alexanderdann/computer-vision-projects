"""Implements different plotting functionality.

This code was generated in assistance with an LLM
"""  # noqa: INP001

import operator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.cm import viridis
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator


def plot_gradient_flow(gradient_history: list[dict]) -> Figure:
    """Visualize how gradients behave during training.

    Args:
        gradient_history: list of dictionaries with gradient norms for each
            layer over time. Each dictionary should have layer names as keys
            and gradient norms as values

    Returns:
        The figure object for further customization or saving

    """
    title = "Gradient Flow During Training"

    # Set up the figure with a custom style
    plt.style.use("default")
    fig = plt.figure(figsize=(12, 7))
    fig.patch.set_facecolor("white")

    # Extract layer names and organize data
    layer_names = list(gradient_history[0].keys())
    epochs = np.arange(1, len(gradient_history) + 1)

    # Create a color palette from viridis colormap
    colors = [viridis(i / len(layer_names)) for i in range(len(layer_names))]

    # Main plot for gradient flow over time
    ax1 = fig.add_subplot(111)

    for i, layer_name in enumerate(layer_names):
        layer_grads = [epoch_data[layer_name] for epoch_data in gradient_history]
        ax1.plot(
            epochs,
            layer_grads,
            "o-",
            linewidth=1,
            markersize=0,
            color=colors[i],
            label=layer_name,
            alpha=0.8,
        )

    # Add a horizontal line at y=0
    ax1.axhline(y=0, color="k", linestyle="--", alpha=0.3)

    # Customize the plot
    ax1.set_xlabel("Training Iterations", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Average Gradient Magnitude", fontsize=12, fontweight="bold")
    ax1.set_title(title, fontsize=16, fontweight="bold", pad=20)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Add legend with custom styling
    _ = ax1.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        fancybox=True,
        shadow=True,
        fontsize=10,
    )

    # Add a grid
    ax1.grid(visible=True, linestyle="--", alpha=0.7)

    # Add text describing what this plot shows
    plt.figtext(
        0.02,
        0.02,
        "This plot shows the average magnitude of gradients for "
        "each layer during training.\n"
        "Spikes may indicate learning or instability, while gradual "
        "decline suggests convergence.",
        fontsize=9,
        style="italic",
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    return fig


def plot_validation_accuracy(
    accuracies: list[float],
    loss_values: list[float] | None = None,
    title="Model Performance",
) -> Figure:
    """Plot validation accuracy over epochs with optional training loss.

    Args:
        accuracies: List of accuracy values for each epoch
        loss_values: Optional list of training loss values
        title: Title for the plot

    Returns:
        The figure object for further customization or saving

    """
    plt.style.use("default")
    fig = plt.figure(figsize=(12, 9))
    fig.patch.set_facecolor("white")

    # Create a GridSpec layout for our plots
    gs = (
        gridspec.GridSpec(2, 1, height_ratios=[3, 2])
        if loss_values
        else gridspec.GridSpec(1, 1)
    )

    # Main accuracy plot
    ax1 = fig.add_subplot(gs[0])
    epochs = np.arange(1, len(accuracies) + 1)

    # Plot accuracy with gradient fill
    _ = ax1.plot(
        epochs,
        accuracies,
        "o-",
        linewidth=3,
        markersize=8,
        color="#1f77b4",
        label="Validation Accuracy",
    )
    ax1.fill_between(epochs, 0, accuracies, alpha=0.2, color="#1f77b4")

    # Add moving average for trend line
    window_size = min(5, len(accuracies))
    if window_size > 1:
        moving_avg = np.convolve(
            accuracies,
            np.ones(window_size) / window_size,
            mode="valid",
        )
        padded_avg = np.pad(moving_avg, (window_size - 1, 0), "edge")
        ax1.plot(
            epochs,
            padded_avg,
            "--",
            linewidth=2,
            color="#ff7f0e",
            label=f"{window_size}-Epoch Moving Average",
        )

    # Find and mark the best accuracy
    best_epoch = np.argmax(accuracies) + 1
    best_acc = np.max(accuracies)
    ax1.plot(best_epoch, best_acc, "o", markersize=12, color="green", alpha=0.8)
    ax1.annotate(
        f"Best: {best_acc:.2%}",
        xy=(best_epoch, best_acc),
        xytext=(best_epoch - 0.5, best_acc + 0.1),
        fontsize=12,
        fontweight="bold",
    )

    # Customize the plot
    ax1.set_xlabel("Epochs", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Validation Accuracy", fontsize=12, fontweight="bold")
    ax1.set_title(title, fontsize=16, fontweight="bold", pad=20)
    ax1.set_ylim([0, max(1.0, max(accuracies) * 1.1)])
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax1.grid(visible=True, linestyle="--", alpha=0.7)

    # Add legend
    ax1.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        fancybox=True,
        shadow=True,
        ncol=2,
        fontsize=12,
    )

    # Add loss subplot if provided
    if loss_values:
        assert len(accuracies) == len(loss_values)  # noqa: S101
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(
            epochs,
            loss_values,
            "o-",
            linewidth=2,
            markersize=6,
            color="#d62728",
            label="Training Loss",
            alpha=0.8,
        )

        # Customize loss subplot
        ax2.set_xlabel("Epochs", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Loss", fontsize=12, fontweight="bold")
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.grid(visible=True, linestyle="--", alpha=0.7)
        ax2.legend(loc="upper right", fancybox=True, shadow=True)

    # Add text describing convergence
    final_acc = accuracies[-1]
    plt.figtext(
        0.02,
        0.02,
        f"Final accuracy: {final_acc:.2%}\n"
        f"Best accuracy: {best_acc:.2%} (epoch {best_epoch})\n"
        f"Improvement from first epoch: {accuracies[-1] - accuracies[0]:.2%}",
        fontsize=10,
        style="italic",
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    return fig


def plot_model_predictions(  # noqa: PLR0914
    images: np.ndarray,
    true_labels: np.ndarray,
    predictions: np.ndarray,
    class_names: list[str],
    num_samples: int,
) -> Figure:
    """Display a grid of sample images with their predictions.

    Args:
        images: Array of validation images (N, H, W, C)
        true_labels: One-hot encoded true labels (N, num_classes)
        predictions: Model predictions, either probabilities or one-hot (N, num_classes)
        class_names: List of class names for each label
        num_samples: Number of samples to display (perfect square recommended)

    Returns:
        The figure object for further customization or saving

    """
    # Convert one-hot to indices if needed
    if len(true_labels.shape) > 1:
        true_labels = np.argmax(true_labels, axis=1)
    if len(predictions.shape) > 1:
        predictions = np.argmax(predictions, axis=1)

    # Determine grid size
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    num_samples = min(num_samples, len(images))

    # Create figure with custom styling
    plt.style.use("default")
    fig = plt.figure(figsize=(15, 15))
    fig.suptitle(
        "Model Predictions on Validation Data",
        fontsize=20,
        fontweight="bold",
        y=0.95,
    )

    # Calculate metrics for the subplot
    correct = np.sum(predictions[:num_samples] == true_labels[:num_samples])
    accuracy = correct / num_samples

    # Add a subtitle with accuracy
    plt.figtext(
        0.5,
        0.91,
        f"Accuracy on these samples: {accuracy:.2%} ({correct}/{num_samples} correct)",
        ha="center",
        fontsize=14,
        bbox={"facecolor": "orange", "alpha": 0.2, "pad": 5},
    )

    # Create a mapping of which classes are most often confused
    confusion = {}
    for i in range(num_samples):
        if predictions[i] != true_labels[i]:
            pair = (class_names[true_labels[i]], class_names[predictions[i]])
            confusion[pair] = confusion.get(pair, 0) + 1

    # Display images in a grid
    for i in range(num_samples):
        ax = plt.subplot(grid_size, grid_size, i + 1)

        # Handle different image formats
        img = images[i]
        if img.shape[-1] == 1:  # Grayscale image with channel dimension
            img = img.squeeze()

        # Display the image
        gray_scale = 2
        plt.imshow(img, cmap="gray" if len(img.shape) == gray_scale else None)

        # Get prediction details
        pred_label = predictions[i]
        true_label = true_labels[i]
        is_correct = pred_label == true_label

        # Set title color based on correctness
        title_color = "green" if is_correct else "red"
        conf_msg = "" if is_correct else f"\nShould be: {class_names[true_label]}"

        # Set the title
        ax.set_title(
            f"Prediction: {class_names[pred_label]}{conf_msg}",
            color=title_color,
            fontweight="bold",
        )

        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Add a border based on correctness
        border_color = "green" if is_correct else "red"
        [i.set_color(border_color) for i in ax.spines.values()]
        [i.set_linewidth(2) for i in ax.spines.values()]

    # Add information about common misclassifications
    if confusion:
        most_confused = sorted(
            confusion.items(),
            key=operator.itemgetter(1),
            reverse=True,
        )
        confusion_text = "Most common misclassifications:\n"
        for (true_cls, pred_cls), count in most_confused[:3]:
            confusion_text += f"• {true_cls} → {pred_cls}: {count} times\n"

        plt.figtext(
            0.02,
            0.02,
            confusion_text,
            fontsize=12,
            bbox={"facecolor": "white", "alpha": 0.8, "pad": 5},
        )

    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
    return fig
