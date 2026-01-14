#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse


def v_mix(v_slm: float, v_llm: float, y: float, rtt_ms: float, misc_sec_per_tok: float = 0.0) -> float:
    """
    Hybrid effective speed with remote LLM (per-LLM-token RTT cost) and llm_consecutive_tokens=1:

      per_token_time_mix
        = 1/v_slm
        + (1-y) * ( 1/v_llm + rtt )
        + misc

    Notes:
      - v_llm is compute-only steady-state decode speed on LLM host.
      - rtt is round-trip latency paid per LLM token (because we need logits to sample).
    """
    if v_slm <= 0 or v_llm <= 0:
        raise ValueError("v_slm and v_llm must be > 0.")
    if not (0.0 <= y <= 1.0):
        raise ValueError("slm_usage must be in [0, 1].")
    if rtt_ms < 0:
        raise ValueError("rtt_ms must be >= 0.")
    if misc_sec_per_tok < 0:
        raise ValueError("misc_sec_per_tok must be >= 0.")

    r = rtt_ms / 1000.0
    per_tok = (1.0 / v_slm) + (1.0 - y) * ((1.0 / v_llm) + r) + misc_sec_per_tok
    return 1.0 / per_tok


def infer_misc_from_observed(v_slm: float, v_llm: float, y: float, rtt_ms: float, v_obs: float) -> float:
    """
    misc = 1/v_obs - [ 1/v_slm + (1-y)*(1/v_llm + rtt) ]
    """
    if v_obs <= 0:
        raise ValueError("observed speed must be > 0.")
    r = rtt_ms / 1000.0
    base = (1.0 / v_slm) + (1.0 - y) * ((1.0 / v_llm) + r)
    return (1.0 / v_obs) - base


def main():
    ap = argparse.ArgumentParser(
        description="Hybrid speed + speedup vs LLM-only streaming baseline (baseline tok/s = v_llm)."
    )
    ap.add_argument("--v_slm", type=float, required=True, help="SLM decode speed (tok/s)")
    ap.add_argument("--v_llm", type=float, required=True, help="LLM decode speed (tok/s) (streaming baseline)")
    ap.add_argument("--slm_usage", type=float, required=True, help="SLM usage Y in [0,1]")
    ap.add_argument("--rtt_ms", type=float, default=0, help="RTT between hosts (ms), paid per LLM token in hybrid")

    ap.add_argument("--misc_us_per_tok", type=float, default=None,
                    help="Extra overhead per token (microseconds/token). Optional.")
    ap.add_argument("--observed", type=float, default=None,
                    help="Observed hybrid effective speed (tok/s). Optional; infer misc from it.")
    args = ap.parse_args()

    v_slm = args.v_slm
    v_llm = args.v_llm
    y = args.slm_usage
    rtt_ms = args.rtt_ms

    print("Inputs")
    print(f"  v_slm     = {v_slm:.6f} tok/s")
    print(f"  v_llm     = {v_llm:.6f} tok/s  (LLM-only streaming baseline)")
    print(f"  slm_usage = {y:.6f}")
    print(f"  rtt_ms    = {rtt_ms:.3f} ms (paid per LLM token in hybrid)")
    print()

    # Baseline: streaming LLM-only
    v_base = v_llm
    print("Baseline (LLM-only streaming)")
    print(f"  v_base = {v_base:.6f} tok/s")
    print()

    # Theoretical best (misc=0)
    v_best = v_mix(v_slm, v_llm, y, rtt_ms, misc_sec_per_tok=0.0)
    print("Theoretical best hybrid (misc = 0)")
    print(f"  v_mix   = {v_best:.6f} tok/s")
    print(f"  speedup = {v_best / v_base:.6f}x  (vs streaming LLM-only)")
    print()

    # Predicted with provided misc
    if args.misc_us_per_tok is not None:
        misc_s = args.misc_us_per_tok * 1e-6
        v_pred = v_mix(v_slm, v_llm, y, rtt_ms, misc_sec_per_tok=misc_s)
        print("Predicted hybrid with provided misc")
        print(f"  misc    = {misc_s:.9f} s/tok  ({args.misc_us_per_tok:.3f} us/tok)")
        print(f"  v_mix   = {v_pred:.6f} tok/s")
        print(f"  speedup = {v_pred / v_base:.6f}x  (vs streaming LLM-only)")
        print()

    # Observed: report observed speedup and infer misc
    if args.observed is not None:
        v_obs = args.observed
        print("Observed hybrid")
        print(f"  v_mix   = {v_obs:.6f} tok/s")
        print(f"  speedup = {v_obs / v_base:.6f}x  (vs streaming LLM-only)")
        print()

        misc = infer_misc_from_observed(v_slm, v_llm, y, rtt_ms, v_obs)
        print("Inferred misc from observed hybrid")
        print(f"  misc    = {misc:.9f} s/tok  ({misc * 1e6:.3f} us/tok)")
        misc_clamped = max(misc,0.0)
        v_recon = v_mix(v_slm, v_llm, y, rtt_ms, misc_sec_per_tok=misc_clamped)
        print(f"  recon v = {v_recon:.6f} tok/s (misc clamped >= 0)")
        print(f"  recon s = {v_recon / v_base:.6f}x  (vs streaming LLM-only)")


if __name__ == "__main__":
    main()
 
