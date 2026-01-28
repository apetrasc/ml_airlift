import os
import numpy as np


def main():
    # Paths
    input_dir = "/home/smatsubara/documents/airlift/data/experiments/dataset/dropped_data"
    output_dir = "/home/smatsubara/documents/airlift/data/experiments/dataset/nowall"

    x_path = os.path.join(input_dir, "x_train_dropped.npy")
    t_path = os.path.join(input_dir, "t_train_dropped.npy")

    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print(f"[INFO] Loading X from: {x_path}")
    x = np.load(x_path)
    print(f"[INFO] Loading T from: {t_path}")
    t = np.load(t_path)

    # Confirm shapes
    print(f"[INFO] X shape (before): {x.shape}")
    print(f"[INFO] T shape: {t.shape}")

    if x.ndim != 4:
        raise ValueError(f"Expected X to have 4 dimensions (N, C, H, W), but got shape: {x.shape}")

    # Slice W dimension: keep indices 500:2500 (Python slice is [500:2500))
    # This keeps 2000 points in W direction.
    print("[INFO] Slicing W dimension: keeping indices 500:2500")
    x_sliced = x[:, :, :, 500:2500]

    print(f"[INFO] X shape (after): {x_sliced.shape}")

    # Save to output directory
    x_out_path = os.path.join(output_dir, "x_train_nowall.npy")
    t_out_path = os.path.join(output_dir, "t_train_nowall.npy")

    print(f"[INFO] Saving sliced X to: {x_out_path}")
    np.save(x_out_path, x_sliced)

    print(f"[INFO] Saving T (unchanged) to: {t_out_path}")
    np.save(t_out_path, t)

    print("[OK] Finished creating nowall dataset.")


if __name__ == "__main__":
    main()





