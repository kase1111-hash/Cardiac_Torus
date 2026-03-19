"""
03_torus_mapping.py — Map beat pairs onto T² and compute geodesic curvature
True North Research | Cardiac Torus Pipeline

THE CORE MATHEMATICAL CONTRIBUTION

The Cardiac Ramachandran Diagram:
  - For proteins: consecutive residues have angles (φ, ψ) on T²
  - For heartbeats: consecutive beats have features (θ₁, θ₂) on T²

Torus coordinates:
  θ₁ = 2π × (RR_pre - RR_min) / (RR_max - RR_min)     [0, 2π)
  θ₂ = 2π × (RR_post - RR_min) / (RR_max - RR_min)     [0, 2π)

This is a "Poincaré return map on the torus" — each point represents 
the state transition from one inter-beat interval to the next, wrapped 
onto periodic coordinates.

Alternative torus mappings (computed simultaneously):
  Torus A: (RR_pre, RR_post) — interval dynamics
  Torus B: (RR_pre, R_amp_ratio) — interval × morphology  
  Torus C: (RR_ratio, amp_ratio) — normalized dynamics

Geodesic curvature on T²:
  For a curve γ(t) = (θ₁(t), θ₂(t)) on the flat torus (R=r=1):
  κ_g = (θ₁'θ₂'' - θ₂'θ₁'') / (θ₁'² + θ₂'²)^(3/2)

  Computed via Menger curvature of consecutive triples on T²,
  using geodesic distance on T² (shortest path respecting periodicity).

Output: results/torus_curvature.csv
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (RESULTS_DIR, RR_MIN_MS, RR_MAX_MS,
                     AMP_RATIO_MIN, AMP_RATIO_MAX)


def to_angle(value: float, vmin: float, vmax: float) -> float:
    """Map a bounded value to [0, 2π) with clipping."""
    clipped = np.clip(value, vmin, vmax)
    normalized = (clipped - vmin) / (vmax - vmin)
    return 2 * np.pi * normalized


def torus_geodesic_distance(theta1_a, theta2_a, theta1_b, theta2_b):
    """
    Geodesic distance on flat torus T² = [0,2π) × [0,2π).
    
    The flat torus has the property that geodesic distance is just
    the Euclidean distance with periodic boundary conditions:
    
    d(a, b) = sqrt(min(|Δθ₁|, 2π-|Δθ₁|)² + min(|Δθ₂|, 2π-|Δθ₂|)²)
    """
    d1 = np.abs(theta1_a - theta1_b)
    d1 = np.minimum(d1, 2 * np.pi - d1)
    
    d2 = np.abs(theta2_a - theta2_b)
    d2 = np.minimum(d2, 2 * np.pi - d2)
    
    return np.sqrt(d1**2 + d2**2)


def menger_curvature_torus(p1, p2, p3):
    """
    Menger curvature for three consecutive points on the flat torus.
    
    κ = 4A / (|a||b||c|)
    
    where A is the area of the triangle formed by the three points,
    and a, b, c are the side lengths (geodesic distances on T²).
    
    For the flat torus, we unfold to minimize total geodesic distance
    then compute Menger curvature in the covering space.
    """
    # Side lengths (geodesic)
    a = torus_geodesic_distance(p2[0], p2[1], p3[0], p3[1])  # |p2-p3|
    b = torus_geodesic_distance(p1[0], p1[1], p3[0], p3[1])  # |p1-p3|
    c = torus_geodesic_distance(p1[0], p1[1], p2[0], p2[1])  # |p1-p2|
    
    # Avoid degenerate triangles
    if a < 1e-10 or b < 1e-10 or c < 1e-10:
        return 0.0
    
    # Heron's formula for area
    s = (a + b + c) / 2
    area_sq = s * (s - a) * (s - b) * (s - c)
    
    if area_sq <= 0:
        return 0.0  # Collinear points → zero curvature
    
    area = np.sqrt(area_sq)
    kappa = 4 * area / (a * b * c)
    
    return kappa


def compute_signed_curvature_torus(theta1, theta2):
    """
    Compute signed geodesic curvature for a sequence of points on T².
    
    Sign convention: positive = turning left (counterclockwise),
    negative = turning right (clockwise) on the torus.
    
    Uses the unwrapped velocity/acceleration approach:
    κ_g = (v₁ × a₁ - v₂ × a₂) / |v|³
    where × denotes the 2D cross product (scalar).
    """
    n = len(theta1)
    if n < 3:
        return np.zeros(n)
    
    # Unwrap angles to handle periodic boundaries
    # This gives us the "covering space" representation
    dtheta1 = np.diff(theta1)
    dtheta2 = np.diff(theta2)
    
    # Correct for wrapping: if jump > π, subtract 2π; if < -π, add 2π
    dtheta1 = np.where(dtheta1 > np.pi, dtheta1 - 2*np.pi, dtheta1)
    dtheta1 = np.where(dtheta1 < -np.pi, dtheta1 + 2*np.pi, dtheta1)
    dtheta2 = np.where(dtheta2 > np.pi, dtheta2 - 2*np.pi, dtheta2)
    dtheta2 = np.where(dtheta2 < -np.pi, dtheta2 + 2*np.pi, dtheta2)
    
    # Velocities (centered differences using unwrapped deltas)
    # v[i] ≈ (p[i+1] - p[i-1]) / 2
    kappa = np.zeros(n)
    
    for i in range(1, n - 1):
        v1 = dtheta1[i-1] + dtheta1[min(i, len(dtheta1)-1)]  # ~2*velocity
        v2 = dtheta2[i-1] + dtheta2[min(i, len(dtheta2)-1)]
        
        # Acceleration (second difference)
        if i < n - 1:
            a1 = dtheta1[min(i, len(dtheta1)-1)] - dtheta1[i-1]
            a2 = dtheta2[min(i, len(dtheta2)-1)] - dtheta2[i-1]
        else:
            a1, a2 = 0.0, 0.0
        
        speed_sq = v1**2 + v2**2
        if speed_sq < 1e-20:
            kappa[i] = 0.0
            continue
        
        # 2D cross product: v × a = v1*a2 - v2*a1
        cross = v1 * a2 - v2 * a1
        kappa[i] = cross / (speed_sq ** 1.5) * 4  # factor of 4 from centered diff scaling
    
    return kappa


def map_record_to_torus(df_record: pd.DataFrame) -> pd.DataFrame:
    """
    Map a single record's beats onto multiple torus representations
    and compute curvature for each.
    """
    n = len(df_record)
    if n < 3:
        return pd.DataFrame()
    
    rr_pre = df_record['RR_pre_ms'].values
    rr_post = df_record['RR_post_ms'].values
    amp_ratio = df_record['R_amp_ratio'].values
    
    # === TORUS A: (RR_pre, RR_post) — interval dynamics ===
    theta1_A = np.array([to_angle(rr, RR_MIN_MS, RR_MAX_MS) for rr in rr_pre])
    theta2_A = np.array([to_angle(rr, RR_MIN_MS, RR_MAX_MS) for rr in rr_post])
    
    # === TORUS B: (RR_pre, R_amp_ratio) — interval × morphology ===
    theta1_B = theta1_A.copy()
    theta2_B = np.array([to_angle(a, AMP_RATIO_MIN, AMP_RATIO_MAX) for a in amp_ratio])
    
    # === TORUS C: (RR_ratio, amp_ratio) — normalized ===
    rr_ratio = rr_pre / rr_post  # >1 = decelerating, <1 = accelerating
    theta1_C = np.array([to_angle(r, 0.3, 3.0) for r in rr_ratio])
    theta2_C = theta2_B.copy()
    
    # Compute Menger curvature (unsigned) for each torus
    kappa_A = np.zeros(n)
    kappa_B = np.zeros(n)
    kappa_C = np.zeros(n)
    
    for i in range(1, n - 1):
        p1_A = (theta1_A[i-1], theta2_A[i-1])
        p2_A = (theta1_A[i],   theta2_A[i])
        p3_A = (theta1_A[i+1], theta2_A[i+1])
        kappa_A[i] = menger_curvature_torus(p1_A, p2_A, p3_A)
        
        p1_B = (theta1_B[i-1], theta2_B[i-1])
        p2_B = (theta1_B[i],   theta2_B[i])
        p3_B = (theta1_B[i+1], theta2_B[i+1])
        kappa_B[i] = menger_curvature_torus(p1_B, p2_B, p3_B)
        
        p1_C = (theta1_C[i-1], theta2_C[i-1])
        p2_C = (theta1_C[i],   theta2_C[i])
        p3_C = (theta1_C[i+1], theta2_C[i+1])
        kappa_C[i] = menger_curvature_torus(p1_C, p2_C, p3_C)
    
    # Compute signed curvature for Torus A (primary)
    kappa_signed_A = compute_signed_curvature_torus(theta1_A, theta2_A)
    
    # Build output
    result = df_record.copy()
    result['theta1_A'] = np.round(theta1_A, 6)
    result['theta2_A'] = np.round(theta2_A, 6)
    result['theta1_B'] = np.round(theta1_B, 6)
    result['theta2_B'] = np.round(theta2_B, 6)
    result['theta1_C'] = np.round(theta1_C, 6)
    result['theta2_C'] = np.round(theta2_C, 6)
    result['kappa_A'] = np.round(kappa_A, 6)
    result['kappa_B'] = np.round(kappa_B, 6)
    result['kappa_C'] = np.round(kappa_C, 6)
    result['kappa_signed_A'] = np.round(kappa_signed_A, 6)
    result['RR_ratio'] = np.round(rr_ratio, 4)
    
    return result


def main():
    print("=" * 60)
    print("Step 03: Torus Mapping & Geodesic Curvature")
    print("=" * 60)
    
    # Load beat features
    csv_path = RESULTS_DIR / 'beat_features.csv'
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run 02_extract_beats.py first.")
        sys.exit(1)
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} beats from {df['record'].nunique()} records")
    
    # Process each record
    all_results = []
    for rec_id, group in df.groupby('record'):
        group = group.sort_values('beat_idx').reset_index(drop=True)
        result = map_record_to_torus(group)
        if len(result) > 0:
            all_results.append(result)
            
            # Quick stats
            ka = result['kappa_A']
            valid = ka[ka > 0]
            if len(valid) > 0:
                print(f"  [{rec_id}] {len(result):5d} beats  "
                      f"κ_A: median={np.median(valid):.3f}  "
                      f"max={np.max(valid):.3f}  "
                      f"mean={np.mean(valid):.3f}")
    
    # Concatenate and save
    df_out = pd.concat(all_results, ignore_index=True)
    out_path = RESULTS_DIR / 'torus_curvature.csv'
    df_out.to_csv(out_path, index=False)
    
    # Summary statistics
    print()
    print(f"Total beats with torus coordinates: {len(df_out):,}")
    print()
    
    # Curvature by AAMI class
    print("Curvature (κ_A) by AAMI class:")
    print("-" * 50)
    for cls in ['N', 'S', 'V', 'F', 'Q']:
        subset = df_out[df_out['aami_class'] == cls]['kappa_A']
        valid = subset[subset > 0]
        if len(valid) > 0:
            print(f"  {cls}: n={len(valid):6,}  "
                  f"median={np.median(valid):.4f}  "
                  f"mean={np.mean(valid):.4f}  "
                  f"P95={np.percentile(valid, 95):.4f}")
    
    print()
    
    # Curvature by torus type
    for torus, col in [('A (RR×RR)', 'kappa_A'), 
                        ('B (RR×Amp)', 'kappa_B'),
                        ('C (ratio×Amp)', 'kappa_C')]:
        valid = df_out[col][df_out[col] > 0]
        print(f"Torus {torus}: median κ = {np.median(valid):.4f}, "
              f"mean = {np.mean(valid):.4f}")
    
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
