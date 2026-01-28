#!/usr/bin/env python3
"""
Visualize SimpleCNNReal2D model architecture.
Creates a clear diagram showing the model structure.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

def visualize_simplecnn_real2d_architecture(
    in_channels=2,
    hidden=64,
    out_dim=6,
    use_residual=True,
    resize_hw=None,
    dropout_rate=0.0,
    save_path='simplecnn_real2d_architecture.png'
):
    """
    Create a visual diagram of SimpleCNNReal2D architecture.
    
    Args:
        in_channels: Number of input channels
        hidden: Hidden dimension (base number of channels)
        out_dim: Output dimension
        use_residual: Whether to use residual blocks
        resize_hw: Optional resize dimensions (h, w)
        dropout_rate: Dropout rate
        save_path: Path to save the figure
    """
    fig = plt.figure(figsize=(16, 10 if use_residual else 8))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12 if use_residual else 6)
    ax.axis('off')
    
    # Colors
    color_input = '#E3F2FD'
    color_conv = '#BBDEFB'
    color_residual = '#90CAF9'
    color_pool = '#64B5F6'
    color_fc = '#42A5F5'
    color_output = '#2196F3'
    
    y_start = 11 if use_residual else 5
    y_pos = y_start
    x_center = 5
    
    # Title
    title = f"SimpleCNNReal2D Architecture\n"
    title += f"(in_channels={in_channels}, hidden={hidden}, out_dim={out_dim}, "
    title += f"use_residual={use_residual})"
    ax.text(x_center, y_pos + 0.5, title, ha='center', va='top', 
            fontsize=14, fontweight='bold')
    
    y_pos -= 0.8
    
    # Input
    input_box = FancyBboxPatch((x_center - 1.5, y_pos - 0.3), 3, 0.6,
                               boxstyle="round,pad=0.1", 
                               facecolor=color_input, edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    input_text = f"Input\n[B, {in_channels}, H, W]"
    ax.text(x_center, y_pos, input_text, ha='center', va='center', fontsize=10, fontweight='bold')
    y_pos -= 0.8
    
    # Optional Pre-resize
    if resize_hw is not None:
        arrow = FancyArrowPatch((x_center, y_pos + 0.3), (x_center, y_pos - 0.3),
                               arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
        ax.add_patch(arrow)
        y_pos -= 0.5
        
        prepool_box = FancyBboxPatch((x_center - 1.5, y_pos - 0.3), 3, 0.6,
                                    boxstyle="round,pad=0.1",
                                    facecolor=color_pool, edgecolor='black', linewidth=2)
        ax.add_patch(prepool_box)
        prepool_text = f"AdaptiveAvgPool2d\n({resize_hw[0]}, {resize_hw[1]})"
        ax.text(x_center, y_pos, prepool_text, ha='center', va='center', fontsize=9)
        y_pos -= 0.8
    
    if use_residual:
        # Residual Architecture
        # Initial Conv
        arrow = FancyArrowPatch((x_center, y_pos + 0.3), (x_center, y_pos - 0.3),
                               arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
        ax.add_patch(arrow)
        y_pos -= 0.5
        
        initial_conv_box = FancyBboxPatch((x_center - 2, y_pos - 0.4), 4, 0.8,
                                         boxstyle="round,pad=0.1",
                                         facecolor=color_conv, edgecolor='black', linewidth=2)
        ax.add_patch(initial_conv_box)
        initial_conv_text = "Initial Conv Block\nConv2d({}→{}, k=7, s=2) + BN + ReLU + MaxPool2d(k=3, s=2)".format(
            in_channels, hidden)
        ax.text(x_center, y_pos, initial_conv_text, ha='center', va='center', fontsize=9)
        y_pos -= 1.0
        
        # Residual Blocks
        residual_layers = [
            (hidden, hidden, 1, "Layer1"),
            (hidden, 2 * hidden, 2, "Layer2"),
            (2 * hidden, 4 * hidden, 2, "Layer3"),
            (4 * hidden, 4 * hidden, 1, "Layer4")
        ]
        
        for in_ch, out_ch, stride, layer_name in residual_layers:
            arrow = FancyArrowPatch((x_center, y_pos + 0.3), (x_center, y_pos - 0.3),
                                   arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
            ax.add_patch(arrow)
            y_pos -= 0.5
            
            # Residual block box
            res_box = FancyBboxPatch((x_center - 2.5, y_pos - 0.5), 5, 1.0,
                                    boxstyle="round,pad=0.1",
                                    facecolor=color_residual, edgecolor='black', linewidth=2)
            ax.add_patch(res_box)
            
            # Draw residual connection
            if in_ch == out_ch and stride == 1:
                # Skip connection (identity)
                skip_arrow = FancyArrowPatch((x_center + 2.8, y_pos), (x_center + 2.8, y_pos - 1.0),
                                            arrowstyle='->', mutation_scale=15, linewidth=1.5, 
                                            color='red', linestyle='--', alpha=0.7)
                ax.add_patch(skip_arrow)
                ax.text(x_center + 3.2, y_pos - 0.5, "skip", ha='left', va='center', 
                       fontsize=8, color='red', style='italic')
            
            res_text = f"{layer_name}: ResidualBlock2D\n"
            res_text += f"Conv2d({in_ch}→{out_ch}, k=3, s={stride}) + BN + ReLU\n"
            res_text += f"Conv2d({out_ch}→{out_ch}, k=3, s=1) + BN"
            if dropout_rate > 0:
                res_text += f" + Dropout2d({dropout_rate})"
            res_text += "\n+ ReLU"
            ax.text(x_center, y_pos, res_text, ha='center', va='center', fontsize=8)
            y_pos -= 1.2
        
        # Global Average Pooling
        arrow = FancyArrowPatch((x_center, y_pos + 0.3), (x_center, y_pos - 0.3),
                               arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
        ax.add_patch(arrow)
        y_pos -= 0.5
        
        gap_box = FancyBboxPatch((x_center - 1.5, y_pos - 0.3), 3, 0.6,
                                boxstyle="round,pad=0.1",
                                facecolor=color_pool, edgecolor='black', linewidth=2)
        ax.add_patch(gap_box)
        gap_text = f"Global Average Pooling\n[B, {4 * hidden}, H', W'] → [B, {4 * hidden}, 1, 1]"
        ax.text(x_center, y_pos, gap_text, ha='center', va='center', fontsize=9)
        y_pos -= 0.8
        
        # Flatten
        arrow = FancyArrowPatch((x_center, y_pos + 0.3), (x_center, y_pos - 0.3),
                               arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
        ax.add_patch(arrow)
        y_pos -= 0.5
        
        flatten_text = f"Flatten\n[B, {4 * hidden}, 1, 1] → [B, {4 * hidden}]"
        ax.text(x_center, y_pos, flatten_text, ha='center', va='center', fontsize=9, style='italic')
        y_pos -= 0.6
        
        # Head (FC layers)
        arrow = FancyArrowPatch((x_center, y_pos + 0.3), (x_center, y_pos - 0.3),
                               arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
        ax.add_patch(arrow)
        y_pos -= 0.5
        
        head_box = FancyBboxPatch((x_center - 2, y_pos - 0.5), 4, 1.0,
                                 boxstyle="round,pad=0.1",
                                 facecolor=color_fc, edgecolor='black', linewidth=2)
        ax.add_patch(head_box)
        head_text = f"FC Head\n"
        head_text += f"Dropout({dropout_rate}) → Linear({4 * hidden}→{2 * hidden}) → ReLU\n"
        head_text += f"Dropout({dropout_rate}) → Linear({2 * hidden}→{out_dim})"
        ax.text(x_center, y_pos, head_text, ha='center', va='center', fontsize=9)
        y_pos -= 1.0
        
    else:
        # Simple Architecture
        arrow = FancyArrowPatch((x_center, y_pos + 0.3), (x_center, y_pos - 0.3),
                               arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
        ax.add_patch(arrow)
        y_pos -= 0.5
        
        features_box = FancyBboxPatch((x_center - 2, y_pos - 0.5), 4, 1.0,
                                     boxstyle="round,pad=0.1",
                                     facecolor=color_conv, edgecolor='black', linewidth=2)
        ax.add_patch(features_box)
        features_text = f"Features\n"
        features_text += f"Conv2d({in_channels}→{hidden}, k=3) + BN + ReLU\n"
        features_text += f"Conv2d({hidden}→{hidden // 2}, k=3) + BN + ReLU"
        ax.text(x_center, y_pos, features_text, ha='center', va='center', fontsize=9)
        y_pos -= 1.0
        
        # Global Average Pooling
        arrow = FancyArrowPatch((x_center, y_pos + 0.3), (x_center, y_pos - 0.3),
                               arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
        ax.add_patch(arrow)
        y_pos -= 0.5
        
        gap_box = FancyBboxPatch((x_center - 1.5, y_pos - 0.3), 3, 0.6,
                                boxstyle="round,pad=0.1",
                                facecolor=color_pool, edgecolor='black', linewidth=2)
        ax.add_patch(gap_box)
        gap_text = f"Global Average Pooling\n[B, {hidden // 2}, H', W'] → [B, {hidden // 2}, 1, 1]"
        ax.text(x_center, y_pos, gap_text, ha='center', va='center', fontsize=9)
        y_pos -= 0.8
        
        # Flatten
        arrow = FancyArrowPatch((x_center, y_pos + 0.3), (x_center, y_pos - 0.3),
                               arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
        ax.add_patch(arrow)
        y_pos -= 0.5
        
        flatten_text = f"Flatten\n[B, {hidden // 2}, 1, 1] → [B, {hidden // 2}]"
        ax.text(x_center, y_pos, flatten_text, ha='center', va='center', fontsize=9, style='italic')
        y_pos -= 0.6
        
        # Head (FC layer)
        arrow = FancyArrowPatch((x_center, y_pos + 0.3), (x_center, y_pos - 0.3),
                               arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
        ax.add_patch(arrow)
        y_pos -= 0.5
        
        head_box = FancyBboxPatch((x_center - 1.5, y_pos - 0.3), 3, 0.6,
                                 boxstyle="round,pad=0.1",
                                 facecolor=color_fc, edgecolor='black', linewidth=2)
        ax.add_patch(head_box)
        head_text = f"Linear({hidden // 2}→{out_dim})"
        ax.text(x_center, y_pos, head_text, ha='center', va='center', fontsize=9)
        y_pos -= 0.8
    
    # Output
    arrow = FancyArrowPatch((x_center, y_pos + 0.3), (x_center, y_pos - 0.3),
                           arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow)
    y_pos -= 0.5
    
    output_box = FancyBboxPatch((x_center - 1.5, y_pos - 0.3), 3, 0.6,
                               boxstyle="round,pad=0.1",
                               facecolor=color_output, edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    output_text = f"Output\n[B, {out_dim}]"
    ax.text(x_center, y_pos, output_text, ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=color_input, edgecolor='black', label='Input'),
        mpatches.Patch(facecolor=color_conv, edgecolor='black', label='Convolution'),
        mpatches.Patch(facecolor=color_residual, edgecolor='black', label='Residual Block'),
        mpatches.Patch(facecolor=color_pool, edgecolor='black', label='Pooling'),
        mpatches.Patch(facecolor=color_fc, edgecolor='black', label='Fully Connected'),
        mpatches.Patch(facecolor=color_output, edgecolor='black', label='Output'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Architecture diagram saved to: {save_path}")
    plt.close()


def main():
    """Generate architecture diagrams for both residual and simple architectures."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize SimpleCNNReal2D architecture')
    parser.add_argument('--in_channels', type=int, default=2, help='Number of input channels')
    parser.add_argument('--hidden', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--out_dim', type=int, default=6, help='Output dimension')
    parser.add_argument('--use_residual', action='store_true', default=True, help='Use residual blocks')
    parser.add_argument('--no_residual', action='store_true', help='Use simple architecture')
    parser.add_argument('--resize_hw', type=int, nargs=2, default=None, help='Resize dimensions [h, w]')
    parser.add_argument('--dropout_rate', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--output', type=str, default='simplecnn_real2d_architecture.png', 
                       help='Output file path')
    
    args = parser.parse_args()
    
    use_residual = args.use_residual and not args.no_residual
    resize_hw = tuple(args.resize_hw) if args.resize_hw else None
    
    visualize_simplecnn_real2d_architecture(
        in_channels=args.in_channels,
        hidden=args.hidden,
        out_dim=args.out_dim,
        use_residual=use_residual,
        resize_hw=resize_hw,
        dropout_rate=args.dropout_rate,
        save_path=args.output
    )
    
    # Also generate simple architecture if residual was requested
    if use_residual:
        simple_output = args.output.replace('.png', '_simple.png')
        visualize_simplecnn_real2d_architecture(
            in_channels=args.in_channels,
            hidden=args.hidden,
            out_dim=args.out_dim,
            use_residual=False,
            resize_hw=resize_hw,
            dropout_rate=args.dropout_rate,
            save_path=simple_output
        )


if __name__ == "__main__":
    main()



