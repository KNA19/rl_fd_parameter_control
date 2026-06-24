# FR-Only SARL Experiment Setting

## Purpose

This document freezes the experimental setting for the Fruchterman-Reingold-only SARL parameter-control experiments.

## Algorithm

Fruchterman-Reingold force-directed layout.

## RL Agent

Single-agent PPO.

The agent controls algorithm parameters only. It does not directly move graph nodes.

## State

Full state:

```text
s_t = [G_h, G_e, P_t, A_t, ΔA_t, D_t, C_t, H_t]