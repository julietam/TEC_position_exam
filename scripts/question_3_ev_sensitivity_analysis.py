#!/usr/bin/env python3
"""
Question 3: Fermi Problem and Sensitivity Analysis

Estimate the impact on global CO2 emissions if 50% of the world's population adopted EVs.
Simulate at the country level and identify where reductions would be largest.

Inputs:
- outputs/processed_world_bank_data.csv (from Question 1 pipeline)

Outputs (to outputs/question_3/):
- ev_impact_by_country.csv
- ev_global_summary.json
- ev_country_rankings.png
- ev_sensitivity_ranges.png

Method summary (Fermi-style, documented in the Markdown report too):
1) Baseline transport emissions: use CO2_emissions_transport_Mt. If missing, infer from total with
   income/urbanization-based defaults.
2) Vehicle ownership (vehicles per 1,000 people) is estimated from GDP per capita using
   a piecewise heuristic reflective of global patterns.
3) EV adoption mapping: 50% of population adopting EVs translates to EV share of Vehicle-Km Travelled (VKT)
   as EV_share = min(1, 0.5 / ownership_fraction), where ownership_fraction = vehicles_per_1000/1000.
   This caps at 100% electrification in low-ownership countries (reasonable under Fermi logic).
4) Emissions intensity per km:
   - ICE vehicles: 180 gCO2/km (global average of modern fleet)
   - EVs: depends on grid carbon intensity, proxied from fossil fuel energy share.
     Grid EF (gCO2/kWh) ≈ 50 + 7.5 * fossil_fuel_share_percent  (≈50–800 g/kWh).
     EV uses ≈0.18 kWh/km ⇒ EV_gpkm = 0.18 * Grid_EF (≈9–144 g/km).
5) Annual km per vehicle depends on development level (per-capita GDP):
   - Low: 3,000 km, Lower-Middle: 5,000 km, Upper-Middle: 8,000 km, High: 12,000 km
   This affects internal consistency checks; core result leverages reported transport emissions first.
6) Reduction factor for electrified VKT: (ICE_gpkm - EV_gpkm) / ICE_gpkm, clipped [0,1].
7) Country reduction = transport_emissions * EV_share * reduction_factor (Mt CO2).

Note: This is a transparent Fermi model intended for directional insight and sensitivity analysis,
not precise forecasting.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8')

DATA_PATH = 'outputs/processed_world_bank_data.csv'
OUT_DIR = 'outputs/question_3'
os.makedirs(OUT_DIR, exist_ok=True)

ICE_G_PER_KM = 180.0  # gCO2/km
EV_KWH_PER_KM = 0.18  # kWh/km


def infer_vehicle_ownership_per_1000(gdp_pc: float) -> float:
    """Heuristic vehicles per 1,000 people as a function of GDP per capita (USD)."""
    if pd.isna(gdp_pc):
        return 200.0
    if gdp_pc < 3000:
        return 50.0
    if gdp_pc < 10000:
        return 150.0
    if gdp_pc < 30000:
        return 400.0
    if gdp_pc < 60000:
        return 600.0
    return 750.0


def grid_ef_g_per_kwh(fossil_pct: float, renewable_pct: float) -> float:
    """Proxy grid emission factor (gCO2/kWh) from energy mix.
    Base: 50 g/kWh for very low-fossil grids; add 7.5 g/kWh per fossil percentage point.
    """
    if pd.isna(fossil_pct):
        return 450.0  # neutral default
    ef = 50.0 + 7.5 * np.clip(fossil_pct, 0, 100)
    # very renewable-heavy grids can be slightly lower than the base
    if not pd.isna(renewable_pct) and renewable_pct > 60:
        ef = max(30.0, ef - 20.0)
    return ef


def annual_km_per_vehicle(gdp_pc: float) -> float:
    if pd.isna(gdp_pc):
        return 7000.0
    if gdp_pc < 3000:
        return 3000.0
    if gdp_pc < 10000:
        return 5000.0
    if gdp_pc < 30000:
        return 8000.0
    return 12000.0


def estimate_transport_share(total_mt: float, gdp_pc: float, urb_pct: float) -> float:
    """If transport emissions missing, infer share of total based on development level."""
    if pd.isna(total_mt) or total_mt <= 0:
        return 0.0
    # baseline by income tier
    if pd.isna(gdp_pc):
        base = 0.15
    elif gdp_pc < 3000:
        base = 0.10
    elif gdp_pc < 10000:
        base = 0.12
    elif gdp_pc < 30000:
        base = 0.16
    else:
        base = 0.20
    # adjust slightly by urbanization
    if not pd.isna(urb_pct):
        base = base + (urb_pct - 50.0) * 0.0005  # +/- 2.5pp for 100% vs 0%
    return float(np.clip(base, 0.05, 0.30))


def main():
    df = pd.read_csv(DATA_PATH)

    # Latest year per country
    latest = df.loc[df.groupby('country')['date'].idxmax()].copy()

    # Baseline transport emissions (Mt)
    transport_mt = latest['CO2_emissions_transport_Mt'].copy() if 'CO2_emissions_transport_Mt' in latest else pd.Series(np.nan, index=latest.index)

    # If missing, estimate from total using heuristic share
    missing_mask = transport_mt.isna()
    if missing_mask.any():
        est_share = latest.apply(lambda r: estimate_transport_share(
            r.get('CO2_emissions_Mt', np.nan),
            r.get('GDP_per_capita_current_USD', np.nan),
            r.get('Urban_population_percent', np.nan)
        ), axis=1)
        transport_mt = transport_mt.fillna(latest['CO2_emissions_Mt'] * est_share)

    # Ownership fraction
    latest['vehicles_per_1000'] = latest['GDP_per_capita_current_USD'].apply(infer_vehicle_ownership_per_1000)
    latest['ownership_fraction'] = np.clip(latest['vehicles_per_1000'] / 1000.0, 0.01, 0.95)

    # EV share of VKT given 50% of population adopt EVs
    latest['ev_share_vkt'] = np.minimum(1.0, 0.5 / latest['ownership_fraction'])

    # Grid emission factor and EV g/km
    latest['grid_ef_g_per_kwh'] = latest.apply(
        lambda r: grid_ef_g_per_kwh(r.get('Fossil_fuel_energy_consumption_percent', np.nan),
                                    r.get('Renewable_energy_consumption_percent', np.nan)), axis=1)
    latest['ev_g_per_km'] = EV_KWH_PER_KM * latest['grid_ef_g_per_kwh']

    # Reduction factor for electrified VKT
    latest['reduction_factor'] = np.clip((ICE_G_PER_KM - latest['ev_g_per_km']) / ICE_G_PER_KM, 0.0, 1.0)

    # Country reduction (Mt) = transport emissions * EV share * reduction factor
    latest['transport_emissions_Mt'] = transport_mt
    latest['ev_reduction_Mt'] = latest['transport_emissions_Mt'] * latest['ev_share_vkt'] * latest['reduction_factor']

    # Percent reduction relative to total emissions
    latest['total_emissions_Mt'] = latest.get('CO2_emissions_Mt', pd.Series(np.nan, index=latest.index))
    latest['pct_of_total_reduced'] = 100.0 * (latest['ev_reduction_Mt'] / latest['total_emissions_Mt'])

    # Global summary
    global_transport = float(latest['transport_emissions_Mt'].sum())
    global_reduction = float(latest['ev_reduction_Mt'].sum())
    global_total = float(latest['total_emissions_Mt'].sum())
    summary = {
        'global_transport_emissions_Mt': round(global_transport, 3),
        'global_total_emissions_Mt': round(global_total, 3),
        'global_ev_reduction_Mt': round(global_reduction, 3),
        'global_ev_reduction_percent_of_transport': round(100.0 * global_reduction / global_transport, 2) if global_transport > 0 else None,
        'global_ev_reduction_percent_of_total': round(100.0 * global_reduction / global_total, 2) if global_total > 0 else None,
        'assumptions': {
            'ICE_g_per_km': ICE_G_PER_KM,
            'EV_kWh_per_km': EV_KWH_PER_KM,
            'grid_ef_formula': '50 + 7.5 * fossil_percent (gCO2/kWh), -20 if renewables > 60% (min 30)',
            'ev_share_mapping': 'EV_share = min(1, 0.5 / ownership_fraction)',
            'ownership_from_gdp': 'piecewise heuristic based on GDP per capita',
            'km_per_vehicle_by_income': {
                'low': 3000, 'lower_middle': 5000, 'upper_middle': 8000, 'high': 12000
            }
        }
    }

    # Save outputs
    cols_out = [
        'country', 'date', 'transport_emissions_Mt', 'total_emissions_Mt',
        'vehicles_per_1000', 'ownership_fraction', 'ev_share_vkt',
        'grid_ef_g_per_kwh', 'ev_g_per_km', 'reduction_factor',
        'ev_reduction_Mt', 'pct_of_total_reduced', 'GDP_per_capita_current_USD',
        'Urban_population_percent', 'Fossil_fuel_energy_consumption_percent', 'Renewable_energy_consumption_percent'
    ]
    latest[cols_out].sort_values('ev_reduction_Mt', ascending=False).to_csv(
        os.path.join(OUT_DIR, 'ev_impact_by_country.csv'), index=False)

    with open(os.path.join(OUT_DIR, 'ev_global_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Visualization: Top-20 absolute reductions
    top20 = latest.sort_values('ev_reduction_Mt', ascending=False).head(20)
    plt.figure(figsize=(12, 8))
    sns.barplot(data=top20, x='ev_reduction_Mt', y='country', color='teal')
    plt.xlabel('Estimated CO2 Reduction (Mt) - 50% EV adoption scenario')
    plt.ylabel('Country')
    plt.title('Top 20 Countries by Absolute CO2 Reduction (Transport)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'ev_country_rankings.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Simple sensitivity bands: vary fossil share ±15pp and ICE g/km ±20
    def compute_variant(df_in: pd.DataFrame, fossil_delta: float, ice_delta: float) -> pd.Series:
        grid_ef = EV_KWH_PER_KM * (50.0 + 7.5 * np.clip((df_in['Fossil_fuel_energy_consumption_percent'] + fossil_delta).clip(0, 100), 0, 100))
        ev_gpkm = grid_ef
        reduction_factor = np.clip(((ICE_G_PER_KM + ice_delta) - ev_gpkm) / (ICE_G_PER_KM + ice_delta), 0.0, 1.0)
        return df_in['transport_emissions_Mt'] * df_in['ev_share_vkt'] * reduction_factor

    latest['ev_reduction_low'] = compute_variant(latest, fossil_delta=+15, ice_delta=+20)
    latest['ev_reduction_high'] = compute_variant(latest, fossil_delta=-15, ice_delta=-20)

    # Get sensitivity data for top 20 countries
    top20_with_sens = latest.sort_values('ev_reduction_Mt', ascending=False).head(20)
    
    # Plot sensitivity ranges for top 20 baseline
    plt.figure(figsize=(12, 8))
    y = np.arange(len(top20_with_sens))
    base = top20_with_sens['ev_reduction_Mt'].values
    low = top20_with_sens['ev_reduction_low'].values
    high = top20_with_sens['ev_reduction_high'].values
    plt.hlines(y, low, high, color='gray', alpha=0.5)
    plt.plot(base, y, 'o', color='navy')
    plt.yticks(y, top20_with_sens['country'])
    plt.xlabel('Estimated CO2 Reduction (Mt)')
    plt.title('Sensitivity Ranges (Top 20) - Fossil ±15pp, ICE ±20 g/km')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'ev_sensitivity_ranges.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Console summary
    print('=' * 70)
    print('EV Adoption Sensitivity Analysis (50% of population)')
    print('=' * 70)
    print(json.dumps(summary, indent=2))
    print('\nTop 10 countries (Mt reduction):')
    print(latest.sort_values('ev_reduction_Mt', ascending=False)[['country', 'ev_reduction_Mt']].head(10).to_string(index=False))


if __name__ == '__main__':
    main()
